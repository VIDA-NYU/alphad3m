import os
import logging
import itertools
from nltk.grammar import Production, Nonterminal, CFG, is_terminal, is_nonterminal

logger = logging.getLogger(__name__)
BASE_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), '../resource/base_grammar.bnf')
COMPLETE_GRAMMAR_PATH = os.path.join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'complete_grammar.bnf')
TASK_GRAMMAR_PATH = os.path.join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'task_grammar.bnf')


def load_grammar(grammar_path):
    logger.info('Loading grammar in %s' % grammar_path)
    with open(grammar_path) as fin:
        grammar_string = fin.read()

    return CFG.fromstring(grammar_string)


def create_completegrammar(primitives):
    base_grammar = load_grammar(BASE_GRAMMAR_PATH)
    new_productions = []

    for production in base_grammar.productions():
        primitive_type = production.lhs().symbol()
        if primitive_type in primitives:
            new_rhs_list = []
            for token in production.rhs():
                if isinstance(token, str) and token.startswith('primitive_'):
                    new_rhs_list.append(primitives[primitive_type])
                else:
                    new_rhs_list.append([token])
            for new_rhs in itertools.product(*new_rhs_list):
                new_productions.append(Production(production.lhs(), new_rhs))
        else:
            new_productions.append(production)

    complete_grammar = CFG(Nonterminal('S'), new_productions)

    with open(COMPLETE_GRAMMAR_PATH, 'w') as fout:
        fout.write('\n'.join([str(x) for x in complete_grammar.productions()]))

    return complete_grammar


def create_taskgrammar(grammar, task, encoders):
    logger.info('Creating specific grammar for task %s' % task)
    productions = grammar.productions(Nonterminal(task))
    start_token = Nonterminal('S')
    new_productions = []

    for start_production in productions:
        first_token = start_production.rhs()[0]
        if is_nonterminal(first_token) and first_token.symbol().endswith('_TASK'):
            for new_start_production in grammar.productions(first_token):
                new_productions.append(Production(start_token, new_start_production.rhs()))
        else:
            new_productions.append(Production(start_token, start_production.rhs()))

    for production in grammar.productions():
        for new_production in new_productions:
            if production.lhs() in new_production.rhs() and production not in new_productions:
                if production.lhs().symbol() == 'ENCODERS':  # Use encoders only for types of features in the dataset
                    if len(encoders) > 0:
                        new_productions.append(Production(production.lhs(), [Nonterminal(e) for e in encoders]))
                    else:
                        new_productions.append(Production(production.lhs(), ['E']))
                else:
                    new_productions.append(production)

    task_grammar = CFG(start_token, new_productions)

    with open(TASK_GRAMMAR_PATH, 'w') as fout:
        fout.write('\n'.join([str(x) for x in task_grammar.productions()]))

    return task_grammar


def format_grammar(task, primitives, encoders=[]):
    grammar = create_completegrammar(primitives)
    grammar = create_taskgrammar(grammar, task, encoders)
    formatted_grammar = {'NON_TERMINALS': {}, 'TERMINALS': {}, 'RULES': {}, 'RULES_LOOKUP': {}}
    formatted_grammar['START'] = grammar.start().symbol()
    terminals = []

    logger.info('Formating grammar to style of pipeline game')
    for production in grammar.productions():
        non_terminal = production.lhs().symbol()
        production_str = str(production).replace('\'', '')

        formatted_grammar['RULES'][production_str] = len(formatted_grammar['RULES']) + 1

        if non_terminal not in formatted_grammar['NON_TERMINALS']:
            formatted_grammar['NON_TERMINALS'][non_terminal] = len(formatted_grammar['NON_TERMINALS']) + 1

        if non_terminal not in formatted_grammar['RULES_LOOKUP']:
            formatted_grammar['RULES_LOOKUP'][non_terminal] = []
        formatted_grammar['RULES_LOOKUP'][non_terminal].append(production_str)

        for token in production.rhs():
            if is_terminal(token) and token != 'E' and token not in terminals:
                terminals.append(token)

    formatted_grammar['TERMINALS'] = {t: i+len(formatted_grammar['NON_TERMINALS']) for i, t in enumerate(terminals, 1)}
    formatted_grammar['TERMINALS']['E'] = 0  # Special case for the empty symbol

    return formatted_grammar
