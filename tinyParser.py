from tinyTokenizer import Tokenizer, TokenKind
from tinyBlock import tinyBlock, SSA
from copy import deepcopy
from collections import defaultdict
import enum
# from time import sleep

class EmptyPHI:
    def __init__(self, name, prev_id, cur_id):
        self.name = name
        self.prev_id = prev_id
        self.cur_id = cur_id

class tinyParser:
    def __init__(self, codeblock):
        def ssaTypeCastingPurpose() -> list[SSA]:
            return []
        def tinyBlockTypeCastingPurpose() -> list[tinyBlock]:
            return [tinyBlock(0)]
        def emptyPHIsTypeCastingPurpose() -> dict[int, EmptyPHI]:
            return {}
        def functionsTypeCastingPurpose() -> dict[str, int]:
            return {}

        self.SSAs = ssaTypeCastingPurpose()  # list of SSA
        self.tinyBlocks = tinyBlockTypeCastingPurpose()    # list of tinyBlock, with 0 being the block to hold constants
        self.string = codeblock
        # self.val = 0
        self.tokenizer = Tokenizer(codeblock)
        self.cur = self.tokenizer.cur   # current token
        self.curBlock_id = 0            # current block
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
        self.variableModifiedLocation = defaultdict(list)  # {varName : block_id} # it will get reset once phi propagation is complete
        # self.opDomTree = defaultdict(DomTree)
        self.loop_layer = 0         # number that shows the current layer within the loop. Only do a CSE and PHI propagation when this is 0
        self.emptyPHIs = emptyPHIsTypeCastingPurpose()         # {cur_id : emptyPHI}
        self.functions = functionsTypeCastingPurpose()         # {function name : func_block_id}

    def printAllVarTable(self)->None:
        for block_id in range(len(self.tinyBlocks)):
            block = self.tinyBlocks[block_id]
            print(f"block {block_id}")
            for name, ssa in block.varTable.items():
                print(f"{name}: {ssa}")
            print()

    def printSSA(self, SSA_id)->None:
        print(self.SSAs[SSA_id])

    def printBlockSSAs(self, block_id)->None:
        for SSA_id in self.tinyBlocks[block_id].SSA_ids:
            self.printSSA(SSA_id)

    def printAllSSAs(self)->None:
        for ssa in self.SSAs:
            print(ssa)

    def printAllPhis(self)->None:
        for block_id in range(1, len(self.tinyBlocks)):
            block = self.tinyBlocks[block_id]
            for ssa_id in block.SSA_ids:
                ssa = self.SSAs[ssa_id]
                if ssa.op == TokenKind.PHI:
                    print(block.getName(ssa_id), "PHI:", ssa_id, "leftBlock:", ssa.leftBlock, "rightBlock:", ssa.rightBlock)

    def declareVariable(self) -> None:
        name = self.cur.name
        cur_block = self.tinyBlocks[self.curBlock_id]

        if name is None:
            raise SyntaxError(f"Error occurred on index:{self.tokenizer.index}, while declaring variable: identifier name is None: {self.cur}")

        zero_id = self.tinyBlocks[0].getVar(0)
        cur_block.assignVar(name, zero_id)      # initializing variable to 0

    def makeConstant(self, value) -> int:
        """
        does not modify the existing constant's thing
        :return: the SSA_id of the created constant
        """
        constant_block = self.tinyBlocks[0]

        if value is None:
            raise SyntaxError(f"Error occurred on index:{self.tokenizer.index}, while declaring constant: constant value is None: {self.cur}")
        elif type(value) is not int:
            raise SyntaxError(f"Error occurred on index:{self.tokenizer.index}, while declaring constant: constant is not integer: {self.cur}")

        _id = constant_block.getVar(value)
        if _id == -1:   # constant not initialized
            _id = self.makeSSA(self.curBlock_id, TokenKind.CONSTANT, value)
            constant_block.assignVar(value, _id)

        return _id

    def makeBlock(self, prev_blocks: list[int]) -> int:
        """
        Creates tinyBlock, appends to self.tinyBlock and returns the id
        it DOES NOT handle the copying of varTable: you have to do that on your own
        :return: block_id of the created block
        """

        block_id = len(self.tinyBlocks)
        block = tinyBlock(block_id)
        self.tinyBlocks.append(block)

        if len(prev_blocks) > 0:
            prev0 = self.tinyBlocks[prev_blocks[0]]
            block.varTable = deepcopy(prev0.varTable)

            for prev in prev_blocks:
                block.prev.append(prev)

        # create empty header
        self.makeSSA(block_id, TokenKind.EMPTY)

        return block_id

    def get_last_block_id(self, original_block_id):
        """
        :param original_block_id:
        :return: last node/block of the given block id
        """
        block_id = original_block_id
        prev = original_block_id
        while True:
            if block_id is None:
                break

            block = self.tinyBlocks[block_id]
            if block.is_header_block or block.is_if_block:
                prev = block_id
                block_id = block.branching_block
                continue
            else:
                prev = block_id
                block_id = block.fall_through_block
                continue

        return prev

    def getCurSSA_id(self) -> int:
        """
        :raise: SyntaxError when SSA is not initialized variable or when the self.cur is neither identifier nor constant
        :return: SSA_id of self.cur (identifier/constant)
        """
        constant_block = self.tinyBlocks[0]
        cur_block = self.tinyBlocks[self.curBlock_id]

        if self.check(TokenKind.IDENTIFIER):
            return cur_block.getVar(self.cur.name)
        elif self.check(TokenKind.CONSTANT):
            return constant_block.getVar(self.cur.value)
        else:
            raise SyntaxError(f"Error occurred on index:{self.tokenizer.index}, getting SSA_id of current token: token is neither IDENTIFIER nor CONSTANT {self.cur}")

    def returnDominatingSSA(self, block_id:int, SSA_id: int) -> int:
        """
        :param block_id: the block that the SSA resides in
        :param SSA_id: SSA id
        :return: dominating SSA's id, or -1 if no domination is possible
        """
        block = self.tinyBlocks[block_id]
        dominating_id = block.block_id
        to_eliminate = self.SSAs[SSA_id]
        while dominating_id is not None:
            block = self.tinyBlocks[dominating_id]
            # @TODO
            # might need to make the loop more efficient
            for ssa_id in block.SSA_ids:
                ssa = self.SSAs[ssa_id]
                if (ssa.op == to_eliminate.op
                      and ssa.left == to_eliminate.left
                      and ssa.right == to_eliminate.right
                      and ssa.id != to_eliminate.id):
                    return ssa.id
                elif (ssa.op == to_eliminate.op
                      and ssa.op in [TokenKind.PLUS, TokenKind.MULT]
                      and ssa.left == to_eliminate.right
                      and ssa.right == to_eliminate.left
                      and ssa.id != to_eliminate.id):
                    return ssa.id
            dominating_id = block.dominated_by
        return -1

    def CSE(self, block_id, ssa_id):
        """
        CSE: common subexpression elimination
        only do it when we're in the outermost loop
        :param block_id:
        :param ssa_id:
        :return:
        """
        ssa = self.SSAs[ssa_id]
        if self.loop_layer == 0 and ssa.op in [TokenKind.PLUS, TokenKind.MINUS, TokenKind.MULT, TokenKind.DIV, TokenKind.CMP]:
            dom_id = self.returnDominatingSSA(block_id, ssa_id)  # check if there's dominating SSA that has the same left and right (and op, which is implied)
            if dom_id != -1:
                return dom_id
        return -1

    def makeSSA(self, block_id: int,
                op: TokenKind,
                left = None,
                right = None,
                leftBlock = None,
                rightBlock = None,
                name = None,
                paramNum = None) -> int:
        """
        Creates tinyBlock, appends to self.tinyBlock
        :return: SSA_id of newly created SSA
        """
        ssa_id = len(self.SSAs)
        cur_block = self.tinyBlocks[block_id]
        constant_block = self.tinyBlocks[0]
        SSA_obj = SSA(ssa_id, op, left, right, leftBlock, rightBlock, name, paramNum)

        self.SSAs.append(SSA_obj)

        # CSE: common subexpression elimination
        cse = self.CSE(block_id, ssa_id)
        if cse != -1:
            self.SSAs.pop(ssa_id)
            return cse

        # only append if cse don't need to happen currently
        if op == TokenKind.IDENTIFIER:    # No IR for Identifier assignment
            raise SyntaxError("Don't call makeSSA on IDENTIFIER")
        elif op == TokenKind.CONSTANT:
           constant_block.SSA_ids.append(ssa_id)
        elif op == TokenKind.PHI:
            cur_block.SSA_ids.insert(1, ssa_id)
        else:
            cur_block.SSA_ids.append(ssa_id)

        return ssa_id

    def check(self, kind: TokenKind) -> bool:
        return self.cur.kind == kind

    def checkForTerminal(self, terminalName: str):
        try:
            return self.cur.kind in self.first[terminalName]
        except KeyError:
            raise KeyError(f"index:{self.tokenizer.index} - Make sure you spelled the terminal name correctly: {terminalName}")
        
    def next(self) -> None:
        self.cur = self.tokenizer.next()

    def calculateDominator(self, left, right) -> int:
        left_block = self.tinyBlocks[left]
        right_block = self.tinyBlocks[right]

        # n^2
        while left_block.dominated_by is not None:
            rb = right_block
            if left_block.block_id == rb.block_id:
                return left_block.block_id
            while rb.dominated_by is not None:
                rb = self.tinyBlocks[rb.dominated_by]
                if left_block.block_id == rb.block_id:
                    return left_block.block_id
            left_block = self.tinyBlocks[left_block.dominated_by]

        return left_block.block_id

    def getDominanceFrontier(self) -> dict[int, set[int]]:
        """
        :return: returns Dominance frontiers (dict) of the current blocks -> {block_id : set[block_id]}
        """
        df = defaultdict(set)   # {block_id : set[block_id]}
        for block_id in range(1, len(self.tinyBlocks)):
            block = self.tinyBlocks[block_id]
            for prev in block.prev:
                runner = prev
                while runner != block.dominated_by:
                    df[runner].add(block_id)
                    runner = self.tinyBlocks[runner].dominated_by
        return df

    def locationToPlacePhi(self, modified_block_ids: list[int]) -> set[int]:
        """
        :param modified_block_ids: location of block that modified the variable
        :return: location to place phi -> set[block_ids]
        """
        df = self.getDominanceFrontier()
        prev = set()
        for block_id in modified_block_ids:
            prev = prev.union(df[block_id])
        cur = prev.union(*[df[block_id] for block_id in prev])  # nice syntactic sugar

        while prev != cur:
            prev = cur.copy()
            cur = cur.union(*[df[block_id] for block_id in prev])

        return cur

    ### COMPUTATION STARTS
    def computation(self) -> None: # 1st draft clear
        """
        computation = "main" [varDecl] {funcDecl} "{" statSequence "}" "."
        """
        self.makeConstant(0)
        if self.checkForTerminal("computation"): # "main"
            self.next()

            # first blocks init
            self.curBlock_id = self.makeBlock([])  # the first "real code" block
            self.tinyBlocks[0].fall_through_block = self.curBlock_id
            self.makeSSA(block_id = 1, op = TokenKind.FIRSTBLOCK)

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

        # block initialization
        func_block_id = self.makeBlock([])
        func_block = self.tinyBlocks[func_block_id]
        func_block.prev = []

        # memoing current block
        cur_block_id = self.curBlock_id
        self.curBlock_id = func_block_id

        if self.cur.kind == TokenKind.VOID: # ["void"]
            # this means return type is void
            # aka void function
            self.next()
        if self.cur.kind == TokenKind.FUNC: # ["void"] "function"
            self.next()

            if self.cur.kind == TokenKind.IDENTIFIER: # ["void"] "function" ident
                name = self.cur.name
                self.functions[name] = func_block_id
                self.next()
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting identifier")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"void\" or \"function\"")

        # ["void"] "function" ident formalParam
        if self.checkForTerminal("formalParam"):
            self.formalParam()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"(\"")

        if self.cur.kind == TokenKind.SEMICOLON: # ["void"] "function" ident formalParam ";"
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \";\"")

        # ["void"] "function" ident formalParam ";" funcBody
        if self.cur.kind == TokenKind.VAR or self.cur.kind == TokenKind.OPEN_CURLY:
            self.funcBody()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"var\" or \"{{\"")

        #  ["void"] "function" ident formalParam ";" funcBody ";"
        if self.check(TokenKind.SEMICOLON):
            self.curBlock_id = cur_block_id
            self.next()
            return
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \";\"")

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
                else:  # statement { ";" statement } [";"]
                    break
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
        # "let"
        if self.check(TokenKind.LET):
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"let\"")

        # "let" ident
        if self.check(TokenKind.IDENTIFIER):
            name = self.cur.name
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"IDENTIFIER\"")

        # "let" ident "<-"
        if self.check(TokenKind.ASSIGNMENT):
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"<-\"")

        # "let" ident "<-" expression
        if self.checkForTerminal("expression"):
            newVarId = self.expression()
            cur_block = self.tinyBlocks[self.curBlock_id]
            cur_block.assignVar(name, newVarId)
            self.variableModifiedLocation[name].append(self.curBlock_id)

        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['expression']}\"")

    def funcCall(self) -> int: # 1st draft clear TESTED
        """
        "call" ident [ "(" [expression { "," expression } ] ")" ]
                     above part looks complicated, but it's just parameters: call samplefunction (abc, s, 1 + 1), which is optional (this in the bracket)
        call func
        call func ()
        call func (a)
        call func (a, b, c, ...)
        above both are possible
        """

        left = None
        func = 0
        func_name = ""
        params = []

        if self.check(TokenKind.CALL):  # "call"
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"call\"")

        if self.check(TokenKind.IDENTIFIER):   # "call" ident
            func_name = self.cur.name
            self.next()   # identifier gets handled in tokenizer
            # do I need to keep the identifier name to use it later?
        elif self.check(TokenKind.INPUTNUUM):
            left = self.makeSSA(self.curBlock_id, op = TokenKind.INPUTNUUM)
            func = TokenKind.INPUTNUUM
            self.next()
        elif self.check(TokenKind.OUTPUTNEWLINE):
            left = self.makeSSA(self.curBlock_id, op = TokenKind.OUTPUTNEWLINE)
            func = TokenKind.OUTPUTNEWLINE
            self.next()
        elif self.check(TokenKind.OUTPUTNUM):
            left = self.makeSSA(self.curBlock_id, op = TokenKind.OUTPUTNUM)
            func = TokenKind.OUTPUTNUM
            self.next()
            pass
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"IDENTIFIER\"")

        if self.check(TokenKind.OPEN_PARAN):  # "call" ident [ "("
            # the function has a parameter
            self.next()
            if self.checkForTerminal("expression"):  # "call" ident [ "(" [expression
                # parameter collection
                params.append(self.expression())

                while self.check(TokenKind.COMMA):  # "call" ident [ "(" [expression { ","
                    # (a, b, c, ...)
                    self.next()
                    if self.checkForTerminal("expression"):  # "call" ident [ "(" [expression { "," expression } ]
                        params.append(self.expression())
                    else:
                        raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['expression']}\"")
            if self.check(TokenKind.CLOS_PARAN):  # "call" ident [ "(" [expression { "," expression } ] ")" ]
                # closing the parenthesis: could be 'call func()' or 'call func(abc, a, b)' or etc.
                # EXECUTE the function???

                # pre_defined
                if func in [TokenKind.OUTPUTNUM, TokenKind.INPUTNUUM, TokenKind.OUTPUTNEWLINE]:
                    if params:
                        self.SSAs[left].left = params[0]
                else:
                    # function location
                    func_block_id = self.functions[func_name]
                    func_block = self.tinyBlocks[func_block_id]

                    # setpar
                    for num_param in range(1, len(params) + 1):
                        self.makeSSA(self.curBlock_id, op = TokenKind.SETPAR, left = params[num_param - 1], paramNum = num_param)

                    # function call
                    cur_block = self.tinyBlocks[self.curBlock_id]
                    cur_block.function_calls.add(func_block_id)
                    left = self.makeSSA(self.curBlock_id, TokenKind.JSR, func_block.SSA_ids[0])      # ssa_id

                self.next()
            else:
                SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \")\"")
        else:  # the function doesn't have the parameter: 'call func'
            # I don't call next because we're not on the function call anymore
            # we're at the next char of 'call func'
            # EXECUTE the function???

            # pre-defined
            if func in [TokenKind.OUTPUTNUM, TokenKind.INPUTNUUM, TokenKind.OUTPUTNEWLINE]:
                pass
            # @TODO
            # else:  # user_defined
            #     pass

        # self.curBlock_id = next_block.block_id
        return left

    def ifStatement(self): # 1st draft clear
        """
        "if" relation "then" statSequence [ "else" statSequence ] "fi"
        """
        if self.check(TokenKind.IF):  # "if"
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"if\"")

        if self.checkForTerminal("relation"):  # "if" relation
            branching_SSA = self.relation()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['relation']}\"")

        if self.check(TokenKind.THEN):  # "if" relation "then"
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"then\"")

        if self.checkForTerminal("statSequence"):  # "if" relation "then" statSequence
            self.loop_layer += 1

            # creating blocks
            initial_branch = self.tinyBlocks[self.curBlock_id]
            initial_branch.is_if_block = True
            if_block = self.tinyBlocks[self.makeBlock([initial_branch.block_id])]

            # set domination
            if_block.dominated_by = initial_branch.block_id
            initial_branch.dominates.append(if_block.block_id)

            # inheriting the variable Table
            if_block.varTable = deepcopy(initial_branch.varTable)

            # making contents within the if_block
            self.curBlock_id = if_block.block_id
            self.statSequence()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['statSequence']}\"")

        if self.check(TokenKind.ELSE):  # "if" relation "then" statSequence [ "else"
            else_block = self.tinyBlocks[self.makeBlock([initial_branch.block_id])]
            else_block.varTable = deepcopy(initial_branch.varTable)

            # set domination
            else_block.dominated_by = initial_branch.block_id
            initial_branch.dominates.append(else_block.block_id)

            self.next()
            if self.checkForTerminal("statSequence"):  # "if" relation "then" statSequence [ "else" statSequence ]
                # getting the if side last block
                if_side_last_block = self.tinyBlocks[self.get_last_block_id(if_block.block_id)]

                self.curBlock_id = else_block.block_id
                self.statSequence()

                # branch statement
                self.SSAs[branching_SSA].right = else_block.SSA_ids[0]

                # else side
                else_side_last_block = self.tinyBlocks[self.get_last_block_id(else_block.block_id)]

                # join_block initialization
                join_block = self.tinyBlocks[self.makeBlock([if_side_last_block.block_id, else_side_last_block.block_id])]
                dominating_block = self.tinyBlocks[self.calculateDominator(if_side_last_block.block_id, else_side_last_block.block_id)]
                dominating_block.dominates.append(join_block.block_id)
                join_block.dominated_by = dominating_block.block_id

                # branch statement
                self.SSAs[branching_SSA].right = else_block.SSA_ids[0]

                # joining blocks
                else_side_last_block.fall_through_block = join_block.block_id
                if_side_last_block.branching_block = join_block.block_id
                self.makeSSA(if_side_last_block.block_id, TokenKind.BRA, join_block.SSA_ids[0])

                # set branching block's next block
                initial_branch.fall_through_block = if_block.block_id
                initial_branch.branching_block = else_block.block_id

                # set current_block
                self.curBlock_id = join_block.block_id

            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['statSequence']}\"")
        else:  # "if" relation "then" statSequence
            # set join_block on if-side branch
            # we use else_block as join_block
            if_side_last_block = self.tinyBlocks[self.get_last_block_id(if_block.block_id)]

            # when there was no else: fall through block is the join block
            join_block = self.tinyBlocks[self.makeBlock([if_side_last_block.block_id, initial_branch.block_id])]
            dominating_block = self.tinyBlocks[initial_branch.block_id]
            dominating_block.dominates.append(join_block.block_id)
            join_block.dominated_by = dominating_block.block_id

            # set branching block's next block
            initial_branch.fall_through_block = if_block.block_id
            initial_branch.branching_block = join_block.block_id

            # branch statement
            self.SSAs[branching_SSA].right = join_block.SSA_ids[0]

            # bra statement for
            self.makeSSA(if_side_last_block.block_id, TokenKind.BRA, join_block.SSA_ids[0])
            if_side_last_block.branching_block = join_block.block_id

            # set current_block
            self.curBlock_id = join_block.block_id

        if self.check(TokenKind.FI):  # "if" relation "then" statSequence "fi"
            self.loop_layer -= 1

            # create phi function early:
            for name, prev_id in join_block.varTable.items():
                cur_id = self.makeSSA(join_block.block_id, op = TokenKind.PHI, name = name)
                self.emptyPHIs[cur_id] = EmptyPHI(name, prev_id, cur_id)
                join_block.assignVar(name, cur_id)

            # @TODO there's exact same code that does following in the while loop: make this into helper function
            # modify phi and delete empty phis if we've done with outermost layer
            if self.loop_layer == 0:
                for name, blocks in self.variableModifiedLocation.items():
                    blocks_to_place_phi = self.locationToPlacePhi(blocks)
                    for block_id in blocks_to_place_phi:
                        block = self.tinyBlocks[block_id]
                        left = self.tinyBlocks[block.prev[0]].varTable[name]
                        right = self.tinyBlocks[block.prev[1]].varTable[name]
                        leftBlock = block.prev[0]
                        rightBlock = block.prev[1]
                        ssa = self.SSAs[block.getVar(name)]
                        ssa.left = left
                        ssa.right = right
                        ssa.leftBlock = leftBlock
                        ssa.rightBlock = rightBlock
                        self.emptyPHIs.pop(ssa.id)
                self.variableModifiedLocation = defaultdict(list)

                # erase unnecessary PHIs
                eliminated = {}  # {before : after}
                for emptyPHI in self.emptyPHIs.values():
                    for block_id in range(1, len(self.tinyBlocks)):
                        before = emptyPHI.cur_id
                        after = emptyPHI.prev_id
                        name = emptyPHI.name

                        eliminated[before] = after

                        # removing empty phi
                        block = self.tinyBlocks[block_id]
                        try:
                            block.SSA_ids.remove(before)
                        except ValueError:
                            pass

                        # updating old varTable
                        if before == block.getVar(name):
                            block.assignVar(name, after)

                for block_id in range(1, len(self.tinyBlocks)):
                    # transitive property of elimination
                    # 23 -> 16 and 34 -> 23 implies 34 -> 16, and we do not need 34 -> 23
                    new_eliminated = deepcopy(eliminated)
                    for before, after in eliminated.items():
                        while True:
                            try:
                                new_eliminated[before] = new_eliminated[new_eliminated[before]]
                            except KeyError:
                                break
                    eliminated = new_eliminated

                    # updating SSA contents
                    block = self.tinyBlocks[block_id]
                    for before, after in eliminated.items():
                        for ssa_id in block.SSA_ids:
                            ssa = self.SSAs[ssa_id]
                            if ssa.left == before:
                                ssa.left = after
                            if ssa.right == before:
                                ssa.right = after

            # do CSE when we're on the outermost layer
            if self.loop_layer == 0:
                self.do_cse(initial_branch)

            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"fi\"")

    def whileStatement(self): # 1st draft clear
        """
        "while" relation "do" StatSequence "od"
        """
        if self.check(TokenKind.WHILE):  # "while" relation "do" StatSequence "od"
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"while\"")

        if self.checkForTerminal("relation"):  # "while" relation
            # creating initial blocks and join block
            initial_branch = self.tinyBlocks[self.curBlock_id]

            # if the initial_branch is empty(only has the empty ssa (header ssa), just use it as a header block and don't create extra block
            if len(initial_branch.SSA_ids) == 1:
                header_block = initial_branch
            else:
                header_block = self.tinyBlocks[self.makeBlock([initial_branch.block_id])]      # header_block: aka join block
                header_block.dominated_by = initial_branch.block_id
                initial_branch.dominates.append(header_block.block_id)
                self.curBlock_id = header_block.block_id

            # create phi function early:
            for name, prev_id in header_block.varTable.items():
                cur_id = self.makeSSA(header_block.block_id, op = TokenKind.PHI, name=name)
                self.emptyPHIs[cur_id] = EmptyPHI(name, prev_id, cur_id)
                header_block.assignVar(name, cur_id)

            header_block.is_header_block = True
            branching_SSA = self.relation()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['relation']}\"")

        if self.check(TokenKind.DO):  # "while" relation "do"
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"do\"")

        if self.checkForTerminal("statSequence"):  # "while" relation "do" StatSequence
            # creating loop block and else block
            loop_block = self.tinyBlocks[self.makeBlock([header_block.block_id])]      # fall through
            else_block = self.tinyBlocks[self.makeBlock([header_block.block_id])]      # loop condition not satisfied: branching block

            # domination setup
            loop_block.dominated_by = header_block.block_id
            else_block.dominated_by = header_block.block_id
            header_block.dominates.append(loop_block.block_id)
            header_block.dominates.append(else_block.block_id)

            # set initial_branch's next blocks:
            if initial_branch.block_id != header_block.block_id:
                initial_branch.fall_through_block = header_block.block_id
            header_block.fall_through_block = loop_block.block_id
            header_block.branching_block = else_block.block_id

            # branch statement
            self.SSAs[branching_SSA].right = else_block.SSA_ids[0]

            # increment loop-layer
            self.loop_layer += 1

            # making contents within loop_block
            self.curBlock_id = loop_block.block_id
            self.statSequence()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['statSequence']}\"")

        if self.check(TokenKind.OD):  # "while" relation "do" StatSequence "od"
            # looping back to beginning
            loop_side_last_block = self.tinyBlocks[self.get_last_block_id(loop_block.block_id)]
            loop_side_last_block.branching_block = header_block.block_id
            header_block.prev.append(loop_side_last_block.block_id)

            # branch statement
            self.makeSSA(loop_side_last_block.block_id, TokenKind.BRA, header_block.SSA_ids[0])   # headerbranch guaranteed to have SSA because we added empty earlier

            # decrement loop_layer
            self.loop_layer -= 1

            # current_block_id setup
            self.curBlock_id = else_block.block_id

            # propagate phi and if we've done with outermost layer
            if self.loop_layer == 0:
                for name, blocks in self.variableModifiedLocation.items():
                    blocks_to_place_phi = self.locationToPlacePhi(blocks)
                    for block_id in blocks_to_place_phi:
                        block = self.tinyBlocks[block_id]
                        left = self.tinyBlocks[block.prev[0]].varTable[name]
                        right = self.tinyBlocks[block.prev[1]].varTable[name]
                        leftBlock = block.prev[0]
                        rightBlock = block.prev[1]
                        ssa = self.SSAs[block.getVar(name)]
                        ssa.left = left
                        ssa.right = right
                        ssa.leftBlock = leftBlock
                        ssa.rightBlock = rightBlock
                        self.emptyPHIs.pop(ssa.id)
                self.variableModifiedLocation = defaultdict(list)

                # erase unnecessary PHIs
                eliminated = {} # {before : after}
                for emptyPHI in self.emptyPHIs.values():
                    for block_id in range(1, len(self.tinyBlocks)):
                        before = emptyPHI.cur_id
                        after = emptyPHI.prev_id
                        name = emptyPHI.name

                        eliminated[before] = after

                        # removing empty phi
                        block = self.tinyBlocks[block_id]
                        try:
                            block.SSA_ids.remove(before)
                        except ValueError:
                            pass

                        # updating old varTable
                        if before == block.getVar(name):
                            block.assignVar(name, after)

                for block_id in range(1, len(self.tinyBlocks)):
                    # transitive property of elimination
                    # 23 -> 16 and 34 -> 23 implies 34 -> 16, and we do not need 34 -> 23
                    new_eliminated = deepcopy(eliminated)
                    for before, after in eliminated.items():
                        while True:
                            try:
                                new_eliminated[before] = new_eliminated[new_eliminated[before]]
                            except KeyError:
                                break
                    eliminated = new_eliminated

                    # updating SSA contents
                    block = self.tinyBlocks[block_id]
                    for before, after in eliminated.items():
                        for ssa_id in block.SSA_ids:
                            ssa = self.SSAs[ssa_id]
                            if ssa.left == before:
                                ssa.left = after
                            if ssa.right == before:
                                ssa.right = after

            # do CSE when we're on the outermost layer
            if self.loop_layer == 0:
                self.do_cse(header_block)

            # next
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"od\"")

    def do_cse(self, header_block):
        eliminated = {}  # {before: (after, name)}
        # do CSE
        for block_id in range(header_block.block_id, len(self.tinyBlocks)):
            block = self.tinyBlocks[block_id]
            # @TODO make this loop more efficient if possible
            do_while = 1
            while do_while:
                do_while = 0
                for ssa_id in block.SSA_ids:
                    name = block.getName(ssa_id)
                    cse = self.CSE(block_id, ssa_id)
                    if cse != -1:  # need to eliminate
                        if name is not None:
                            block.assignVar(name, cse)
                        block.SSA_ids.remove(ssa_id)
                        do_while = 1
                        eliminated[ssa_id] = (cse, name)
                        break

        # transitive
        for block_id in range(1, len(self.tinyBlocks)):
            # transitive property of elimination
            # 23 -> 16 and 34 -> 23 implies 34 -> 16, and we do not need 34 -> 23
            new_eliminated = deepcopy(eliminated)
            for before, after in eliminated.items():
                while True:
                    try:
                        new_eliminated[before] = new_eliminated[new_eliminated[before][0]]
                    except KeyError:
                        break
            eliminated = new_eliminated

        # update all eliminated SSAs
        for block_id in range(header_block.block_id, len(self.tinyBlocks)):
            block = self.tinyBlocks[block_id]
            for before, after_name in eliminated.items():
                after = after_name[0]
                name = after_name[1]
                # varTable update
                if name is not None and block.getVar(name) == before:
                    block.assignVar(name, after)

                # ssa update
                for ssa_id in block.SSA_ids:
                    ssa = self.SSAs[ssa_id]
                    if ssa.left == before:
                        ssa.left = after
                    if ssa.right == before:
                        ssa.right = after

    def returnStatement(self): # 1st draft clear
        """
        "return" [expression]
        """
        left = None

        if self.check(TokenKind.RETURN):  # "return"
            self.next()

            if self.checkForTerminal("expression"):  # "return" [expression]
                left = self.expression()

            # return statement
            return_ssa_id = self.makeSSA(self.curBlock_id, op = TokenKind.RETURN)
            return_ssa = self.SSAs[return_ssa_id]

            # if there's something to return, we add it to SSA
            if left is not None:
                return_ssa.left = left


            # @TODO do I put branching statement after return? probably not, but I'm not sure
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"return\"")

    def relation(self) -> int: # 1st draft clear
        """
        expression relOP expression
        """
        if self.checkForTerminal("expression"):
            left_expr = self.expression()
            if self.checkForTerminal("relOp"):
                # make blocks

                if self.check(TokenKind.EQUAL):
                    op = TokenKind.BNE
                elif self.check(TokenKind.NOT_EQUAL):
                    op = TokenKind.BEQ
                elif self.check(TokenKind.LT):
                    op = TokenKind.BGE
                elif self.check(TokenKind.LE):
                    op = TokenKind.BGT
                elif self.check(TokenKind.GT):
                    op = TokenKind.BLE
                elif self.check(TokenKind.GE):
                    op = TokenKind.BLT
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting a relation operator")

                self.next()

                if self.checkForTerminal("expression"):
                    right_expr = self.expression()
                    comparison = self.makeSSA(self.curBlock_id, TokenKind.CMP, left_expr, right_expr)
                    return self.makeSSA(self.curBlock_id, op, comparison)   # it doesn't initialize right (where to go)
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
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"(\"")

        num_param = 0
        while self.cur.kind == TokenKind.IDENTIFIER: # "(" [ident
            # get_par setup
            num_param += 1
            param_ssa = self.makeSSA(block_id = self.curBlock_id, op = TokenKind.GETPAR, paramNum = num_param)

            # declaring variable with get_par
            name = self.cur.name
            block = self.tinyBlocks[self.curBlock_id]
            block.assignVar(name, param_ssa)

            self.next()
            if self.cur.kind == TokenKind.COMMA: # "(" [ident {","
                self.next()
                continue
            elif self.cur.kind == TokenKind.CLOS_PARAN: # "(" [ident {"," ident}] ")"
                self.next()
                return
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \",\" or IDENTIFIER")

        # "(" [ident {"," ident}] ")"
        if self.cur.kind == TokenKind.CLOS_PARAN:
            self.next()
            return
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \")\"")

    def varDecl(self): # 1st draft clear
        """
        varDecl = "var" ident { "," ident } ";"
        """
        if self.cur.kind == TokenKind.VAR: # "var"
            self.next()
            while self.cur.kind == TokenKind.IDENTIFIER: # "var" ident {
                self.declareVariable()      # declares the variable with the current cur (initialized to 0)
                self.next()

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


    def expression(self) -> int:  # SSA first draft complete
        """
        term {("+" | "-") term}
        :returns: SSA_id of the expression
        """
        if self.checkForTerminal("term"):  # term
            SSA_id_left = self.term()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['term']}\"")

        if self.check(TokenKind.PLUS) or self.check(TokenKind.MINUS):  # term {("+" | "-")
            while self.check(TokenKind.PLUS) or self.check(TokenKind.MINUS):  # term {("+" | "-")
                op = self.cur.kind
                self.next()
                if self.checkForTerminal("term"):
                    SSA_id_right = self.term()  # term {("+" | "-") term}
                    SSA_id_left = self.makeSSA(self.curBlock_id, op = op, left = SSA_id_left, right = SSA_id_right)
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['term']}\"")

        return SSA_id_left


    def term(self) -> int:  # SSA first draft complete
        """
        factor {("*" | "/") factor}
        :return: SSA_id of the term
        """
        if self.checkForTerminal("factor"):  # factor
            SSA_id_left = self.factor()
            if self.check(TokenKind.MULT) or self.check(TokenKind.DIV):  # factor {("*" | "/")
                while self.check(TokenKind.MULT) or self.check(TokenKind.DIV):   # factor {("*" | "/")
                    op = self.cur.kind
                    self.next()
                    if self.checkForTerminal("factor"):
                        SSA_id_right = self.factor()  # factor {("*" | "/") factor}
                        SSA_id_left = self.makeSSA(self.curBlock_id, op, SSA_id_left, SSA_id_right)
                    else:
                        raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['factor']}\"")
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['factor']}\"")

        return SSA_id_left

    def factor(self) -> int:  # SSA first draft complete
        """
        ident | number | "(" expression ")" | funcCall

        only non-void functions can be used in expressions
        :return: SSA_id of the factor
        """
        left = None
        if self.check(TokenKind.IDENTIFIER):
            left = self.tinyBlocks[self.curBlock_id].getVar(self.cur.name)
            self.next()
        elif self.check(TokenKind.CONSTANT):
            self.makeConstant(self.cur.value)
            left = self.getCurSSA_id()
            self.next()
        elif self.check(TokenKind.OPEN_PARAN):
            self.next()
            if self.checkForTerminal("expression"):
                left = self.expression()
                if self.check(TokenKind.CLOS_PARAN):
                    self.next()
                else:
                    raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \")\"")
            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['expression']}\"")
        elif self.checkForTerminal("funcCall"):
            left = self.funcCall()  # prob gonna be implemented later in the future
        else:
            SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['factor']}\"")

        return left

if __name__ == "__main__":
    print(input())

