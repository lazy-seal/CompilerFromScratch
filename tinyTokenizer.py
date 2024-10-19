from enum import Enum

from collections import defaultdict


# class TokenCategory(enum):
#     """A categorization of tokens, which combines multiple kinds of
#     tokens together when they serve similar purposes."""
#     IDENTIFIER = 1
#     CONSTANT = 2
#     OPERATION = 3

class TokenKind(Enum):
    # EXTRA
    FIRSTBLOCK = 55     # to dodge the issue when the first block becomes the header block (the prev is the loop block and constant block, which causes errors)

    # OPERATIONS (that keep track of dominance tree)
    PLUS = 3            # +
    MINUS = 4           # -
    MULT = 5            # *
    DIV = 6             # /
    CMP = 37            # comparison

    # OPERATIONS (that doesn't keep track dominance tree
    CONSTANT = 2        # >= '0', <= '9'
    END = 38            # end of program
    BRA = 39            # bra y         # branch to y
    BNE = 40            # bne x y       # branch to y on x not equal
    BEQ = 41            # beq x y       # branch to y on x equal
    BLE = 42            # beq x y       # branch to y on x less than or equal
    BLT = 43            # blt x y       # branch to y on x less than
    BGE = 44            # bge x y       # branch to y on x greater than or equal
    BGT = 45            # bgt x y       # branch to y on x greater
    # x = -1            # less than
    # x = 0             # equal
    # x = 1             # greater than
    PHI = 49            # phi function
    EMPTY = 50

    # functions:
    INPUTNUUM = 46      # read a number from the standard input
    OUTPUTNUM = 47      # write a number to the standard input
    OUTPUTNEWLINE = 48  # write a carriage return to the standard input

    # operations (for user Defined functions)
    JSR = 51      # Calling function: # jump to subroutine x, the return value is the result of the operation
    RETURN = 54            # return result x from subroutine (argument ignored for void functions)
    GETPAR = 52
    SETPAR = 53
    # and all getpar and setpars are defined at runtime with enum.auto

    OPEN_PARAN = 12     # (
    CLOS_PARAN = 13     # )
    OPEN_CURLY = 34     # {
    CLOSE_CURLY = 36    # }

    # relOP
    EQUAL = 15          # ==
    NOT_EQUAL = 16      # !=
    LT = 17             # <, less than
    LE = 18             # <=, less than or equal to
    GT = 19             # >, greater than
    GE = 20             # <=, greater than or equal to

    EOF = 14            # End of File / input
    ASSIGNMENT = 9      # <-
    SEMICOLON = 10      # ;
    COMMA = 35          # ,
    PERIOD = 11         # .

    IDENTIFIER = 1      # letter {letter | digit}
    VAR = 7             # "var"
    # COMPUTATION = 8   # "computation"
    LET = 21            # "let"
    CALL = 22           # "call"
    IF = 23             # "if"
    THEN = 24           # "then"
    ELSE = 25           # "else"
    FI = 26             # "fi"
    WHILE = 27          # "while"
    DO = 28             # "do"
    OD = 29             # "od"
    VOID = 31           # "void"
    FUNC = 32           # "function"
    MAIN = 33           # "main"

    def __repr__(self):
        return f'TokenKind.{self.name}'

# class Result:
#     def __init__(self, kind: TokenKind, value = None): #, address: int, regno: int):
#         self.kind = kind        # integer number specified in TokenKind
#         self.value = value      # value if it has any
#         # self.address = address  # address if it is a variable
#         # self.regno = regno      # register number if it is a reg

class Token:
    def __init__(self, kind: TokenKind, name = None, value = None):
        self.kind = kind            # "type" of the Token, illustrated in TokenKind
        self.value = value          # value of constant if it is constant
        self.name = name            # name (str) of identifier if it is identifier

    def __repr__(self):
        kind = self.kind
        value = self.value if self.value is not None else "None"
        return f"tinyToken = {{kind: {kind} value: {value}}}"

class Tokenizer:
    """Stores and reads input, turns them into tokens"""
    def __init__(self, user_input: str):
        self.user_input = user_input
        self.index = 0
        self.identifiers = defaultdict(lambda : -1) # str name : int id (index of memory)
        self.cur = self.tokenize()

    def next(self):
        self.cur = self.tokenize()
        return self.cur

    def increment(self):
        if self.index < len(self.user_input):
            self.index += 1
            if self.index >= len(self.user_input) - 1:
                return '\0'
            ch = self.user_input[self.index]
            return ch

    def tokenize(self) -> Token:
        """reads a single token from the user_input
        @TODO complete all conditions"""
        # ch = self.increment()
        if self.index >= len(self.user_input):
            ch = '\0'
        else:
            ch = self.user_input[self.index]

        while ch.isspace():
            ch = self.increment()

        if 'a' <= ch <= 'z':
            return self.make_identifier(ch)
        elif '0' <= ch <= '9':
            return self.create_constant(ch)
        elif ch == '+':
            self.increment()
            return Token(TokenKind.PLUS)
        elif ch == '-':
            self.increment()
            return Token(TokenKind.MINUS)
        elif ch == '*':
            self.increment()
            return Token(TokenKind.MULT)
        elif ch == '/':
            self.increment()
            return Token(TokenKind.DIV)
        elif ch == ';':
            self.increment()
            return Token(TokenKind.SEMICOLON)
        elif ch == '.':
            self.increment()
            return Token(TokenKind.PERIOD)
        elif ch == '(':
            self.increment()
            return Token(TokenKind.OPEN_PARAN)
        elif ch == ')':
            self.increment()
            return Token(TokenKind.CLOS_PARAN)
        elif ch == ',':
            self.increment()
            return Token(TokenKind.COMMA)
        elif ch == '{':
            self.increment()
            return Token(TokenKind.OPEN_CURLY)
        elif ch == '}':
            self.increment()
            return Token(TokenKind.CLOSE_CURLY)
        elif ch == '=':
            if self.increment() == '=':
                self.increment()
                return Token(TokenKind.EQUAL)
            else:
                raise SyntaxError(f"Invalid use of {ch} on {self.cur} at position of {self.index} when expecting \"=\"")
        elif ch == '!':  # check for "!="
            if self.increment() == '=':
                self.increment()
                return Token(TokenKind.NOT_EQUAL)
            else:
                raise SyntaxError(f"Invalid use of {ch} on {self.cur} at position of {self.index} when expecting \"=\"")
        elif ch == '<':  # check for  <-
            ch = self.increment()
            if ch == '-':
                self.increment()
                return Token(TokenKind.ASSIGNMENT)
            elif self.increment() == '=':
                self.increment()
                return Token(TokenKind.LE)
            else:
                return Token(TokenKind.LT)
        elif ch == '>':
            if self.increment() == '=':
                self.increment()
                return Token(TokenKind.GE)
            else:
                return Token(TokenKind.GT)
        elif ch == '\0':
            return Token(TokenKind.EOF)
        elif 'A' <= ch <= 'Z':
            return self.make_predefined_function(ch)
        else:
            raise SyntaxError(f"Invalid use of {ch} on {self.cur} at position of {self.index}")

    def make_predefined_function(self, ch) -> Token:
        """parsing pre-defined functions:
        InputNum
        OutputNum
        OutputNewLine
        """
        name = ""
        while ('a' <= ch <= 'z' or
               '0' <= ch <= '9' or
               'A' <= ch <= 'Z'):
            name += ch
            ch = self.increment()

        if name == "InputNum":
            return Token(TokenKind.INPUTNUUM)
        elif name == "OutputNewLine":
            return Token(TokenKind.OUTPUTNEWLINE)
        elif name == "OutputNum":
            return Token(TokenKind.OUTPUTNUM)
        else:
            raise SyntaxError(f"Invalid use of {ch} on {self.cur} at position of {self.index} while expecting pre-defined function")

    def make_identifier(self, ch) -> Token:
        """Declares identifier in self.identifiers,
        initializes it to None
        """
        name = ""
        while 'a' <= ch <= 'z' or '0' <= ch <= '9':
            name += ch
            ch = self.increment()

        if name == "var":
            return Token(TokenKind.VAR)
        elif name == "main":
            return Token(TokenKind.MAIN)
        elif name == "let":
            return Token(TokenKind.LET)
        elif name == "call":
            return Token(TokenKind.CALL)
        elif name == "if":
            return Token(TokenKind.IF)
        elif name == "then":
            return Token(TokenKind.THEN)
        elif name == "else":
            return Token(TokenKind.ELSE)
        elif name == "fi":
            return Token(TokenKind.FI)
        elif name == "while":
            return Token(TokenKind.WHILE)
        elif name == "do":
            return Token(TokenKind.DO)
        elif name == "od":
            return Token(TokenKind.OD)
        elif name == "return":
            return Token(TokenKind.RETURN)
        elif name == "void":
            return Token(TokenKind.VOID)
        elif name == "function":
            return Token(TokenKind.FUNC)


        # if self.identifiers[name] ==- 1:    # if the variable haven't been initialized
        #     self.memory.append(0)        # initializing with 0 first, when declared
        #     self.identifiers[name] = len(self.memory) - 1

        return Token(TokenKind.IDENTIFIER, name=name)

    def create_constant(self, ch) -> Token:
        """Returns a Constant Token"""
        num = ""
        while '0' <= ch <= '9':
            num += ch
            ch = self.increment()
        return Token(TokenKind.CONSTANT, value = int(num))

if __name__ == "__main__":
    sample_input1 = "computation var abc <- 3; i + 4."
    t1 = Tokenizer(sample_input1)
    while t1.cur.kind != TokenKind.EOF:
        # mult is not getting printed : FIX IT
        print(t1.cur)
        t1.next()
    print(t1.cur)
    print(t1.identifiers)

    # Do we check for cases like 123abc? is it allowed?
    # assume no syntax error