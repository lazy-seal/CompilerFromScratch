from tinyTokenizer import Tokenizer, TokenKind
from tinyBlock import tinyBlock, SSA, DomTree

"""
@TODO: assignment, funcCall, ifStatement, whileStatement, returnStatement
"""


class tinyParser:
    def __init__(self, string):
        self.SSAs = []
        self.tinyBlocks = []
        self.string = string
        # self.val = 0
        self.tokenizer = Tokenizer(string)
        self.cur = self.tokenizer.cur   # current token
        self.curBlock = None            # current block
        self.first = {
            "computation": (TokenKind.MAIN,),
            "funcBody": (TokenKind.VAR,),
            "formalParam": (TokenKind.OPEN_PARAN,),
            "funcDecl":(TokenKind.VOID, TokenKind.FUNC),
            "varDecl": (TokenKind.VAR,),
            "statSequence": (TokenKind.LET, TokenKind.CALL, TokenKind.IF, TokenKind.WHILE, TokenKind.RETURN),
            "statement": (TokenKind.LET, TokenKind.CALL, TokenKind.IF, TokenKind.WHILE, TokenKind.RETURN),
            "returnStatement": (TokenKind.RETURN,),
            "whileStatement": (TokenKind.WHILE,),
            "ifStatement": (TokenKind.IF,),
            "funcCall": (TokenKind.CALL,),
            "assignment": (TokenKind.LET,),
            "relation": (TokenKind.IDENTIFIER, TokenKind.CONSTANT, TokenKind.OPEN_PARAN, TokenKind.CALL),
            "expression": (TokenKind.IDENTIFIER, TokenKind.CONSTANT, TokenKind.OPEN_PARAN, TokenKind.CALL),
            "term": (TokenKind.IDENTIFIER, TokenKind.CONSTANT, TokenKind.OPEN_PARAN, TokenKind.CALL),
            "factor": (TokenKind.IDENTIFIER, TokenKind.CONSTANT, TokenKind.OPEN_PARAN, TokenKind.CALL),
            "relOp": (TokenKind.EQUAL, TokenKind.NOT_EQUAL, TokenKind.LT, TokenKind.LE, TokenKind.GT, TokenKind.GE)
        }

    def makeBlock(self) -> int:
        """Creates tinyBlock, appends to self.tinyBlock and returns the id"""
        _id = len(self.tinyBlocks)
        self.tinyBlocks.append(tinyBlock(_id))
        self.curBlock = _id
        return _id

    def check(self, kind: TokenKind) -> bool:
        return self.cur.kind == kind

    def checkForTerminal(self, terminalName: str):
        try:
            return self.cur.kind in self.first[terminalName]
        except KeyError:
            raise KeyError(f"index:{self.tokenizer.index} - Make sure you spelled the terminal name correctly: {terminalName}")
        
    def next(self) -> None:
        self.cur = self.tokenizer.next()
    
    def computation(self) -> None: # 1st draft clear
        """
        computation = "main" [varDecl] {funcDecl} "{" statSequence "}" "."
        """

        self.makeBlock()  # the initial constant block
        if self.checkForTerminal("computation"): # "main"
            self.next()
            self.makeBlock()  # the first "real code" block

            if self.checkForTerminal("varDecl"): # "main" [varDecl]
                self.varDecl()
            if self.checkForTerminal("funcDecl") : # "main" [varDecl] {funcDecl}
                self.funcDecl()

            if self.check(TokenKind.OPEN_CURLY): # "main" [varDecl] {funcDecl} "{"
                self.next()

                if self.checkForTerminal("statSequence"): # "main" [varDecl] {funcDecl} "{" statSequence
                    self.statSequence()
                    if self.check(TokenKind.CLOSE_CURLY): # "main" [varDecl] {funcDecl} "{" statSequence "}"
                        self.next()
                        if self.check(TokenKind.PERIOD): # "main" [varDecl] {funcDecl} "{" statSequence "}" "."
                            return
                        else:
                            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \".\"")
                    else:
                        raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"}}\"")
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"let\" or \"call\" or \"if\" or \"while\" or \"return\"")

        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - The first token should be \"main\", not {self.cur}")

    def funcDecl(self): # 1st draft clear
        """
        funcDecl = ["void"] "function" ident formalParam ";" funcBody ";"
        """
        if self.cur.kind == TokenKind.VOID: # ["void"]
            # this means return type is void
            # aka void function
            self.next()
        if self.cur.kind == TokenKind.FUNC: # ["void"] "function"
            self.next()

            if self.cur.kind == TokenKind.IDENTIFIER: # ["void"] "function" ident
                self.next() # identifier declaration is already handled by tokenizer
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting identifier")

            if self.checkForTerminal("formalParam"): # ["void"] "function" ident formalParam
                self.formalParam()
                if self.cur.kind == TokenKind.SEMICOLON: # ["void"] "function" ident formalParam ";"
                    self.next()
                    if self.cur.kind == TokenKind.VAR or self.cur.kind == TokenKind.OPEN_CURLY: # ["void"] "function" ident formalParam ";" funcBody
                        self.funcBody()
                        if self.check(TokenKind.SEMICOLON): # ["void"] "function" ident formalParam ";" funcBody ";"
                            self.next()
                            return
                        else:
                            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \";\"")
                    else:
                        raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"var\" or \"{{\"")
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \";\"")
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"(\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"void\" or \"function\"")

    def funcBody(self): # 1st draft clear
        """
        funcBody = [varDecl] "{" [statSequence] "}"
        """
        if self.checkForTerminal("varDecl"): # [varDecl]
            self.varDecl()
        if self.cur.kind == TokenKind.OPEN_CURLY: # [varDecl] "{"
            self.next()
            if self.checkForTerminal("statSequence"): # [varDecl] "{" [statSequence]
                self.statSequence()
            if self.check(TokenKind.CLOSE_CURLY): # [varDecl] "{" [statSequence] "}"
                self.next()
                return
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"}}\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"{{\"")


    def statSequence(self): # ist draft clear
        """
        statSequence = statement { ";" statement } [";"]
        """
        if self.checkForTerminal("statement"): # statement
            self.statement()
            while self.check(TokenKind.SEMICOLON): # statement { ";"
                self.next()
                if self.checkForTerminal("statement"):  # statement { ";" statement }
                    self.statement()
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"let\" or \"call\" or \"if\" or \"while\" or \"return\"")
            return # statement { ";" statement } [";"]
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"let\" or \"call\" or \"if\" or \"while\" or \"return\"")

    def statement(self): # 1st draft clear
        """
        statement = assignment | funcCall | ifStatement | whileStatement | returnStatement
        """
        if self.checkForTerminal("assignment"): # assignment
            self.assignment()
        elif self.checkForTerminal("funcCall"): # assignment | funcCall
            self.funcCall()
        elif self.checkForTerminal("ifStatement"): # assignment | funcCall | ifStatement
            self.ifStatement()
        elif self.checkForTerminal("whileStatement"): # assignment | funcCall | ifStatement | whileStatement
            self.whileStatement()
        elif self.checkForTerminal("returnStatement"): # assignment | funcCall | ifStatement | whileStatement | returnStatement
            self.returnStatement()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"let\" or \"call\" or \"if\" or \"while\" or \"return\"")

    def assignment(self): # 1st draft clear
        """
        "let" ident "<-" expression
        """
        if self.check(TokenKind.LET):  # "let"
            self.next()
            if self.check(TokenKind.IDENTIFIER):  # "let" ident
                self.next()  # identifier gets handled in tokenizer
                if self.check(TokenKind.ASSIGNMENT):  # "let" ident "<-"
                    self.next()
                    if self.checkForTerminal("expression"):  # "let" ident "<-" expression
                        self.expression()
                    else:
                        raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['expression']}\"")
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"<-\"")
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"IDENTIFIER\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"let\"")

    def funcCall(self): # 1st draft clear TESTED
        """
        "call" ident [ "(" [expression { "," expression } ] ")" ]
                     above part looks complicated, but it's just parameters: call samplefunction (abc, s, 1 + 1), which is optional (this in the bracket)
        call func
        call func ()
        call func (a)
        call func (a, b, c, ...)
        above both are possible
        """
        if self.check(TokenKind.CALL):  # "call"
            self.next()
            if self.check(TokenKind.IDENTIFIER):   # "call" ident
                self.next()   # identifier gets handled in tokenizer
                if self.check(TokenKind.OPEN_PARAN):  # "call" ident [ "("
                    # the function has a parameter
                    self.next()
                    if self.checkForTerminal("expression"):  # "call" ident [ "(" [expression
                        self.expression()
                        while self.check(TokenKind.COMMA):  # "call" ident [ "(" [expression { ","
                            # (a, b, c, ...)
                            # need some way to accumulate the identifier of the function call separately as a local stack
                            self.next()
                            if self.checkForTerminal("expression"):  # "call" ident [ "(" [expression { "," expression } ]
                                self.expression()
                            else:
                                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['expression']}\"")
                    if self.check(TokenKind.CLOS_PARAN):  # "call" ident [ "(" [expression { "," expression } ] ")" ]
                        # closing the paranthesis: could be 'call func()' or 'call func(abc, a, b)' or etc.
                        self.next()
                        pass  # EXECUTE the function???
                    else:
                        SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \")\"")
                else:  # the function doesn't have the parameter: 'call func'
                    # I don't call next because we're not on the function call anymore
                    # we're at the next char of 'call func'
                    pass  # EXECUTE the function???
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"IDENTIFIER\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"call\"")

    def ifStatement(self): # 1st draft clear
        """
        "if" relation "then" statSequence [ "else" statSequence ] "fi"
        """
        if self.check(TokenKind.IF):  # "if"
            self.next()
            if self.checkForTerminal("relation"):  # "if" relation
                self.relation()
                if self.check(TokenKind.THEN):  # "if" relation "then"
                    self.next()
                    if self.checkForTerminal("statSequence"):  # "if" relation "then" statSequence
                        self.statSequence()
                        if self.check(TokenKind.ELSE):  # "if" relation "then" statSequence [ "else"
                            self.next()
                            if self.checkForTerminal("statSequence"):  # "if" relation "then" statSequence [ "else" statSequence ]
                                self.statSequence()
                            else:
                                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['statSequence']}\"")
                        if self.check(TokenKind.FI):  # "if" relation "then" statSequence [ "else" statSequence ] "fi"
                            self.next()
                        else:
                            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"fi\"")
                    else:
                        raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['statSequence']}\"")
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"then\"")
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['relation']}\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"if\"")

    def whileStatement(self): # 1st draft clear
        """
        "while" relation "do" StatSequence "od"
        """
        if self.check(TokenKind.WHILE):  # "while" relation "do" StatSequence "od"
            self.next()
            if self.checkForTerminal("relation"):  # "while" relation
                self.relation()
                if self.check(TokenKind.DO):  # "while" relation "do"
                    self.next()
                    if self.checkForTerminal("statSequence"):  # "while" relation "do" StatSequence
                        self.statSequence()
                        if self.check(TokenKind.OD):  # "while" relation "do" StatSequence "od"
                            self.next()
                        else:
                            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"od\"")
                    else:
                        raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['statSequence']}\"")
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"do\"")
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['relation']}\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"while\"")

    def returnStatement(self): # 1st draft clear
        """
        "return" [expression]
        """
        if self.check(TokenKind.RETURN):  # "return"
            self.next()
            if self.checkForTerminal("expression"):  # "return" [expression]
                self.expression()
            # do something to return
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"return\"")

    def relation(self): # 1st draft clear
        """
        expression relOP expression
        """
        if self.checkForTerminal("expression"):
            self.expression()
            if self.checkForTerminal("relOp"):
                self.next()  # relation operators are given by the tokenizer
                # I'll prob add if statements for all different relOPs later
                if self.checkForTerminal("expression"):
                    self.expression()
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['expression']}\"")
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['relOP']}\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['expression']}\"")

    def formalParam(self): # 1st draft clear
        """
        formalParam = "(" [ident {"," ident}] ")"
        """
        if self.cur.kind == TokenKind.OPEN_PARAN: # "("
            self.next()
            while self.cur.kind == TokenKind.IDENTIFIER: # "(" [ident
                # do something
                self.next()
                if self.cur.kind == TokenKind.COMMA: # "(" [ident {","
                    self.next()
                    continue
                elif self.cur.kind == TokenKind.CLOS_PARAN: # "(" [ident {"," ident}] ")"
                    self.next()
                    return
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \",\" or IDENTIFIER")
            if self.cur.kind == TokenKind.CLOS_PARAN: # "(" [ident {"," ident}] ")"
                self.next()
                return
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \")\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"(\"")

    def varDecl(self): # 1st draft clear
        """
        varDecl = "var" ident { "," ident } ";"
        """
        if self.cur.kind == TokenKind.VAR: # "var"
            self.next()
            while self.cur.kind == TokenKind.IDENTIFIER: # "var" ident {
                self.next() # identifier declaration is covered by the tokenizer: Tokenizer.make_identifier(self, ch)
                if self.cur.kind == TokenKind.COMMA: # "var" ident { ","
                    self.next()
                    continue
                elif self.cur.kind == TokenKind.SEMICOLON:  # "var" ident { "," ident } ";"
                    self.next()
                    break
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \",\" or \";\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"var\"")


    def expression(self):  # need to modify
        """
        term {("+" | "-") term}
        """
        if self.checkForTerminal("term"):  # term
            self.term()
            if self.check(TokenKind.PLUS) or self.check(TokenKind.MINUS):  # term {("+" | "-")
                while self.check(TokenKind.PLUS) or self.check(TokenKind.MINUS):  # term {("+" | "-")
                    if self.check(TokenKind.PLUS):
                        self.next()
                    elif self.check(TokenKind.MINUS):
                        self.next()
                    if self.checkForTerminal("term"):
                        self.term()  # term {("+" | "-") term}
                    else:
                        raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['term']}\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['term']}\"")


    def term(self):  # need to modify
        """
        factor {("*" | "/") factor}
        """
        if self.checkForTerminal("factor"):  # factor
            self.factor()
            if self.check(TokenKind.MULT) or self.check(TokenKind.DIV):  # factor {("*" | "/")
                while self.check(TokenKind.MULT) or self.check(TokenKind.DIV):   # factor {("*" | "/")
                    if self.check(TokenKind.MULT):
                        self.next()
                    elif self.check(TokenKind.DIV):
                        self.next()
                    if self.checkForTerminal("factor"):
                        self.factor()  # factor {("*" | "/") factor}
                    else:
                        raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['factor']}\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['factor']}\"")

    def factor(self):  # need to modify
        """
        ident | number | "(" expression ")" | funcCall

        only non-void functions can be used in expressions
        """
        if self.check(TokenKind.IDENTIFIER):
            self.next()
        elif self.check(TokenKind.CONSTANT):
            self.next()
        elif self.check(TokenKind.OPEN_PARAN):
            self.next()
            if self.checkForTerminal("expression"):
                self.expression()
                if self.check(TokenKind.CLOS_PARAN):
                    self.next()
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \")\"")
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['expression']}\"")
        elif self.checkForTerminal("funcCall"):
            self.funcCall()
        else:
            SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['factor']}\"")




if __name__ == "__main__":
    pass

