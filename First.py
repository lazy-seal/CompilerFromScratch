

firstDict = {
    "computation":("main",),
    "funcBody":("varDecl",),
    "formalParam":("(",),
    "funcDecl":("void", "function"),
    "varDecl":("var",),
    "statSequence":("statement",),
    "statement":("assignment", "funcCall", "ifStatement", "whileStatement", "returnStatement"),
    "returnStatement":("return",),
    "whileStatement":("while",),
    "ifStatement":("if",),
    "funcCall":("call",),
    "assignment":("let",),
    "relation":("expression",),
    "expression":("term",),
    "term":("factor",),
    "factor":("ident", "constant", "(", "funcCall"),
    "relOp":("==", "!=", "<", "<=", ">", ">=")
}
def first(terminalName: str):
    """
    Helps writing parsing functions
    :param terminalName: a name of terminal
    :return: a string of possible first tokens from the given terminalName
    """
    possibleTokens = ""
    firstTuple = firstDict[terminalName]
    for terminal in firstTuple:
        if terminal in firstDict.keys():
            possibleTokens += first(terminal) + ", "
        else:
            possibleTokens += terminal + ", "

    return possibleTokens.removesuffix(", ")

if __name__ == "__main__":
    for name in firstDict.keys():
        print(f"\"{name}\":({first(name)})")


