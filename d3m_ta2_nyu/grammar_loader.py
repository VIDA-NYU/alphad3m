import os
import json
from nltk import Production, Nonterminal, CFG
#from primitive_loader import D3MPrimitiveLoader


BASE_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), '../resource/base_grammar.bnf')
COMPLETE_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), '../resource/complete_grammar.bnf')
TASK_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), '../resource/task_grammar.bnf')


def load_grammar(grammar_path):
    with open(grammar_path) as fin:
        grammar_string = fin.read()

    return CFG.fromstring(grammar_string)


def create_completegrammar():
    #primitives = D3MPrimitiveLoader.get_primitives_info_summarized()
    base_grammar = load_grammar(BASE_GRAMMAR_PATH)
    new_productions = []
    with open(os.path.join(os.path.dirname(__file__), '../resource/primitives_info_sum.json')) as fin:
        primitives = json.load(fin)

    for production in base_grammar.productions():
        primitive_type = production.lhs().symbol()
        if primitive_type in primitives:
            new_rhs = tuple()
            new_rhs_list = []
            for symbol in production.rhs():
                if isinstance(symbol, str) and symbol.startswith('primitive_'):
                    primitive_names = primitives[primitive_type]
                    new_rhs_list = [new_rhs + (pn,) for pn in primitive_names]
                else:
                    new_rhs += (symbol,)
            if len(new_rhs_list) == 0:
                new_rhs_list = [production.rhs()]
            for new_rhs in new_rhs_list:
                new_productions.append(Production(production.lhs(), new_rhs))
        else:
            new_productions.append(production)

    new_grammar = CFG(Nonterminal('S'), new_productions)

    with open(COMPLETE_GRAMMAR_PATH, 'w') as fout:
        fout.write('\n'.join([str(x) for x in new_grammar.productions()]))

    return new_grammar


def create_subgrammar(grammar, task, filters=[]):
    start_element = Nonterminal('S')
    productions_for_task = []

    for initial_production in grammar.productions(Nonterminal(task)):
        productions_for_task.append(Production(start_element, initial_production.rhs()))
        for production in grammar.productions():
            if production.lhs() in initial_production.rhs(): # TODO: if not in filters, then add them
                productions_for_task.append(production)

    sub_grammar = CFG(start_element, productions_for_task)

    with open(TASK_GRAMMAR_PATH, 'w') as fout:
        fout.write('\n'.join([str(x) for x in sub_grammar.productions()]))

    return sub_grammar


def format(grammar):
    formatted = {'NON_TERMINALS':[], 'RULES': {}, 'RULES_LOOKUP': {}}
    formatted['START'] = grammar.start().symbol()
    for production in grammar.productions():
        non_terminal = production.lhs().symbol()
        production_str = str(production).replace('\'', '')

        formatted['RULES'][production_str] = len(formatted['RULES']) + 1

        if non_terminal not in formatted['NON_TERMINALS']:
            formatted['NON_TERMINALS'].append(non_terminal)

        if non_terminal not in formatted['RULES_LOOKUP']:
            formatted['RULES_LOOKUP'][non_terminal] = []
        formatted['RULES_LOOKUP'][non_terminal].append(production_str)

    print(formatted)



grammar = create_completegrammar()
grammar = create_subgrammar(grammar, 'CLASSIFICATION_TASK')
format(grammar)
