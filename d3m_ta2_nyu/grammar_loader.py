import os
from nltk.grammar import Production, Nonterminal, CFG, is_terminal


BASE_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), '../resource/base_grammar.bnf')
COMPLETE_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), '../resource/complete_grammar.bnf')
TASK_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), '../resource/task_grammar.bnf')


def load_grammar(grammar_path):
    with open(grammar_path) as fin:
        grammar_string = fin.read()

    return CFG.fromstring(grammar_string)


def create_completegrammar(primitives):
    base_grammar = load_grammar(BASE_GRAMMAR_PATH)
    new_productions = []

    for production in base_grammar.productions():
        primitive_type = production.lhs().symbol()
        if primitive_type in primitives:
            new_rhs = tuple()
            new_rhs_list = []
            for token in production.rhs():
                if isinstance(token, str) and token.startswith('primitive_'):
                    primitive_names = [x for x in primitives[primitive_type] if x.endswith('.SKlearn')]
                    new_rhs_list = [new_rhs + (pn,) for pn in primitive_names]
                else:
                    new_rhs += (token,)
            if len(new_rhs_list) == 0:
                new_rhs_list = [production.rhs()]
            for new_rhs in new_rhs_list:
                new_productions.append(Production(production.lhs(), new_rhs))
        else:
            new_productions.append(production)

    complete_grammar = CFG(Nonterminal('S'), new_productions)

    with open(COMPLETE_GRAMMAR_PATH, 'w') as fout:
        fout.write('\n'.join([str(x) for x in complete_grammar.productions()]))

    return complete_grammar


def create_taskgrammar(grammar, task, filters=[]):
    start_token = Nonterminal('S')
    new_productions = []

    for initial_production in grammar.productions(Nonterminal(task)):
        new_productions.append(Production(start_token, initial_production.rhs()))

    for initial_production in grammar.productions(Nonterminal(task)):
        for production in grammar.productions():
            if production.lhs() in initial_production.rhs() and production not in new_productions:
                # TODO: filter productions, e.g. augmentation
                new_productions.append(production)

    task_grammar = CFG(start_token, new_productions)

    with open(TASK_GRAMMAR_PATH, 'w') as fout:
        fout.write('\n'.join([str(x) for x in task_grammar.productions()]))

    return task_grammar


def format_grammar(task, primitives):
    grammar = create_completegrammar(primitives)
    grammar = create_taskgrammar(grammar, task)
    formatted_grammar = {'NON_TERMINALS': {}, 'TERMINALS': {}, 'RULES': {}, 'RULES_LOOKUP': {}}
    formatted_grammar['START'] = grammar.start().symbol()
    terminals = []

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

    formatted_grammar['TERMINALS'] = {t: i+len(formatted_grammar['NON_TERMINALS']) for i,t in enumerate(terminals, 1)}
    formatted_grammar['TERMINALS']['E'] = 0  # Special case for the empty symbol

    return formatted_grammar
