from tinyTokenizer import Tokenizer, TokenKind
from tinyBlock import tinyBlock, SSA
from copy import deepcopy

"""
@TODO:
funcCalls (pre-defined)
whileStatement SSA
userDefinedFunctions

consider opDomTree
empty filler statements: keep track of it and check it every time I make SSAs?

branching and empty:

what if, after we're finished with the graph, we fill in the empty blocks, and add the SSAs for branching statements?
"""


class tinyParser:
    def __init__(self, codeblock):
        self.SSAs = []                 # list of SSA
        self.tinyBlocks = [tinyBlock(0)]    # list of tinyBlock, with 0 being the block to hold constants
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

    def printSSA(self, SSA_id)->None:
        print(self.SSAs[SSA_id])

    # def updatePhi(self, block_id, variable_name, left = None, right = None):
    #     """
    #     modify the given block's phi with the given information
    #     recursively call updatePhi on the next block for the phi use
    #     :return:
    #     """
    #     block = self.tinyBlocks[block_id]
    #     # look up vartable for variable_name: it is a phi function
    #     phi_ssa = self.SSAs[block.getVar(variable_name)]
    #
    #     # # if it's not a phi function, we
    #     # if phi_ssa.op != TokenKind.PHI:
    #     #     self.makeSSA()
    #
    #
    #     pass

    def printBlockSSAs(self, block_id)->None:
        for SSA_id in self.tinyBlocks[block_id].SSA_ids:
            self.printSSA(SSA_id)

    def printAllSSAs(self)->None:
        for ssa in self.SSAs:
            print(ssa)
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

    # def assignVariable(self, SSA_id: int) -> None:
    #     name = self.cur.name
    #     cur_block = self.tinyBlocks[self.curBlock_id]
    #
    #     if name is None:
    #         raise SyntaxError(f"Error occurred on index:{self.tokenizer.index}, while assigning variable: identifier name is None: {self.cur}")
    #
    #     cur_block.assignVar(name, SSA_id)      # the variable now represents SSA id

    def makeBlock(self) -> int:
        """
        Creates tinyBlock, appends to self.tinyBlock and returns the id
        it DOES NOT handle the copying of varTable: you have to do that on your own
        :return: block_id of the created block
        """
        block_id = len(self.tinyBlocks)
        self.tinyBlocks.append(tinyBlock(block_id))
        return block_id

    def get_last_block_id(self, original_block_id):
        """
        :param original_block_id:
        :return: last node/block of the given block id
        """
        def travel(visited, block_id, prev):
            if block_id is None:
                return prev
            if block_id == original_block_id or visited[block_id] == 1:
                return prev
            block = self.tinyBlocks[block_id]
            visited[block_id] = 1
            fb = travel(visited, block.fall_through_block, block_id)
            bb = travel(visited, block.branching_block, block_id)
            return fb if fb > bb else bb


            # else:
            #     raise SyntaxError("get_last_block error: traveling weird?")

        block = self.tinyBlocks[original_block_id]
        visited = [0 for _ in range(len(self.tinyBlocks))]
        if type(block) is int:
            print(f"index:{self.tokenizer.index} - Invalid use of {self.cur}")

        fb = travel(visited, block.fall_through_block, original_block_id)
        bb = travel(visited, block.branching_block, original_block_id)

        return fb if fb > bb else bb

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

    def returnDominatingSSA(self, block_id: int, op: TokenKind, SSA_obj: SSA) -> int:
        """
        :param block_id: block_id of wanted domTree
        :param op: operator TokenKind value
        :param SSA_obj: SSA of current statement
        :return: dominating SSA's id, or -1 if no domination is possible
        """
        _id = len(self.SSAs)
        cur_block = self.tinyBlocks[block_id]
        tree = cur_block.opDomTree[op]
        while tree is not None and tree.SSA_id is not None:
            tree_ssa = self.SSAs[tree.SSA_id]
            if tree_ssa.left == SSA_obj.left and tree_ssa.right == SSA_obj.right:
                return tree.SSA_id
            tree = tree.next
        return -1

    def makeSSA(self, block_id: int, op: TokenKind, left = None, right = None) -> int:
        """
        Creates tinyBlock, appends to self.tinyBlock
        :return: SSA_id of newly created SSA
        """
        _id = len(self.SSAs)
        cur_block = self.tinyBlocks[block_id]
        constant_block = self.tinyBlocks[0]
        SSA_obj = SSA(_id, op, left, right)

        if op in [TokenKind.PLUS, TokenKind.MINUS, TokenKind.MULT, TokenKind.DIV, TokenKind.CMP]:
            dom_id = self.returnDominatingSSA(block_id, op, SSA_obj)  # check if there's dominating SSA that has the same left and right (and op, which is implied)
            if dom_id == -1:
                cur_block.insertDomTreeAtTail(op, _id)
            else:
                return dom_id

        if op == TokenKind.IDENTIFIER:    # No IR for Identifier assignment
            raise SyntaxError("Don't call makeSSA on IDENTIFIER")
        elif op == TokenKind.CONSTANT:
           constant_block.SSA_ids.append(_id)
        elif op == TokenKind.PHI:
            cur_block.SSA_ids.insert(0, _id)
        else:
            cur_block.SSA_ids.append(_id)

        self.SSAs.append(SSA_obj)

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

    ### COMPUTATION STARTS
    def computation(self) -> None: # 1st draft clear
        """
        computation = "main" [varDecl] {funcDecl} "{" statSequence "}" "."
        """
        self.makeConstant(0)
        if self.checkForTerminal("computation"): # "main"
            self.next()
            self.curBlock_id = self.makeBlock()  # the first "real code" block
            # self.tinyBlocks[0].next_blocks.append(self.curBlock_id)
            self.tinyBlocks[0].fall_through_block = self.curBlock_id

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
            left = self.statement()
            while self.check(TokenKind.SEMICOLON): # statement { ";"
                self.next()
                if self.checkForTerminal("statement"):  # statement { ";" statement }
                    left = self.statement()
            return left     # statement { ";" statement } [";"]  # do I need this?
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"let\" or \"call\" or \"if\" or \"while\" or \"return\"")

    def statement(self) -> int: # 1st draft clear
        """
        statement = assignment | funcCall | ifStatement | whileStatement | returnStatement
        """
        left = None
        if self.checkForTerminal("assignment"): # assignment
            left = self.assignment()
        elif self.checkForTerminal("funcCall"): # assignment | funcCall
            self.funcCall()
        elif self.checkForTerminal("ifStatement"): # assignment | funcCall | ifStatement
            left = self.ifStatement()
        elif self.checkForTerminal("whileStatement"): # assignment | funcCall | ifStatement | whileStatement
            self.whileStatement()
        elif self.checkForTerminal("returnStatement"): # assignment | funcCall | ifStatement | whileStatement | returnStatement
            self.returnStatement()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"let\" or \"call\" or \"if\" or \"while\" or \"return\"")
        return left

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
            self.next()  # identifier gets handled in tokenizer
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

            # when join block is there, make/modify the phi statement
            if cur_block.join_block is not None or (cur_block.branching_block is not None and cur_block.branching_block < cur_block.block_id):
                join_block = self.tinyBlocks[cur_block.join_block] if cur_block.join_block else cur_block.branching_block
                ssa = self.SSAs[join_block.getVar(name)]
                if ssa.op != TokenKind.PHI:
                    # before = join_block.getVar(name)
                    after = self.makeSSA(join_block.block_id, TokenKind.PHI, ssa.id, newVarId)
                    join_block.assignVar(name, after)
                    print(name, after)
                    # phi propagation
                    self.loop_update(join_block.block_id, name, after)
                else:
                    ssa.right = newVarId


                # and if the join block is while loop, we need to commit: but I think I can do that in the whileStatement

        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['expression']}\"")

        return newVarId

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

        # # setting the necessary blocks:
        # starting_block = self.tinyBlocks[self.curBlock_id]
        # function_block = self.tinyBlocks[self.makeBlock()]
        # next_block = self.tinyBlocks[self.makeBlock()]
        #
        # # varTable copy and opDomTree
        # function_block.varTable = deepcopy(starting_block.varTable)
        # function_block.opDomTree = deepcopy(starting_block.opDomTree)
        # next_block.varTable = deepcopy(starting_block.varTable)
        # next_block.opDomTree = deepcopy(starting_block.opDomTree)
        #
        # # connecting
        # starting_block.fall_through_block = function_block.block_id
        # function_block.fall_through_block = next_block.block_id
        # self.curBlock_id = function_block.block_id

        left = None
        func = 0
        params = []

        if self.check(TokenKind.CALL):  # "call"
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"call\"")

        if self.check(TokenKind.IDENTIFIER):   # "call" ident
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
                if func == TokenKind.OUTPUTNUM:
                    self.SSAs[left].left = params[0]

                # @TODO
                # else:  # user defined

                self.next()
            else:
                SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \")\"")
        else:  # the function doesn't have the parameter: 'call func'
            # I don't call next because we're not on the function call anymore
            # we're at the next char of 'call func'
            # EXECUTE the function???

            # pre-defined
            if func == TokenKind.INPUTNUUM:
                pass
            elif func == TokenKind.OUTPUTNEWLINE:
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
            # creating blocks
            initial_branch = self.tinyBlocks[self.curBlock_id]
            if_block = self.tinyBlocks[self.makeBlock()]
            else_block = self.tinyBlocks[self.makeBlock()]

            # make join block and set the next block of last blocks
            join_block = self.tinyBlocks[self.makeBlock()]

            # inheriting the variable Table and opDomTree
            if_block.varTable = deepcopy(initial_branch.varTable)
            if_block.opDomTree = deepcopy(initial_branch.opDomTree)
            else_block.varTable = deepcopy(initial_branch.varTable)
            else_block.opDomTree = deepcopy(initial_branch.opDomTree)
            join_block.varTable = deepcopy(initial_branch.varTable)
            join_block.opDomTree = deepcopy(initial_branch.opDomTree)

            # making contents within the if_block
            self.curBlock_id = if_block.block_id
            self.statSequence()
            # if not if_block.SSA_ids:    # if the statSequence only had assignments (didn't produce SSA, we add empty one to guide)
            #     self.makeSSA(if_block.block_id, TokenKind.EMPTY)
            # self.SSAs[branching_SSA].right = if_block.SSA_ids[0]
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['statSequence']}\"")

        if self.check(TokenKind.ELSE):  # "if" relation "then" statSequence [ "else"
            self.next()
            if self.checkForTerminal("statSequence"):  # "if" relation "then" statSequence [ "else" statSequence ]
                self.curBlock_id = else_block.block_id
                self.statSequence()

                if not else_block.SSA_ids:  # if the statSequence only had assignments (didn't produce SSA, we add empty one to guide through)
                    self.makeSSA(else_block.block_id, TokenKind.EMPTY)
                self.SSAs[branching_SSA].right = else_block.SSA_ids[0]

                # if side
                if_side_last_block = self.tinyBlocks[self.get_last_block_id(if_block.block_id)]

                # else side
                else_side_last_block = self.tinyBlocks[self.get_last_block_id(else_block.block_id)]

                # if_side_last_block.next_blocks.append(join_block.block_id)
                else_side_last_block.fall_through_block = join_block.block_id
                # else_side_last_block.next_blocks.append(join_block.block_id)

                # # inherit varTable from last blocks, and create phi function if needed
                # for varName, var_id in if_side_last_block.varTable.items():
                #     join_block.varTable[varName] = var_id
                # for varName, var_id in else_side_last_block.varTable.items():
                #     if join_block.getVar(varName) == -1 or join_block.getVar(varName) == var_id:
                #         # either variable is new, or they have the same id
                #         join_block.varTable[varName] = var_id
                #     else: # has a conflict: need a phi function
                #         join_block.varTable[varName] = self.makeSSA(join_block.block_id, TokenKind.PHI, if_side_last_block.getVar(varName), else_side_last_block.getVar(varName))


                if not join_block.SSA_ids:  # if the statSequence only had assignments (didn't produce SSA, we add empty one to guide)
                    self.makeSSA(join_block.block_id, TokenKind.EMPTY)

                if not else_block.SSA_ids:    # if the statSequence only had assignments (didn't produce SSA, we add empty one to guide)
                    self.makeSSA(else_block.block_id, TokenKind.EMPTY)
                self.SSAs[branching_SSA].right = else_block.SSA_ids[0]

                # bra statement for
                if_side_last_block.branching_block = join_block.block_id
                self.makeSSA(if_side_last_block.block_id, TokenKind.BRA, join_block.SSA_ids[0])

                # set branching block's next block
                # branching_block.next_blocks.append(if_block.block_id)
                # branching_block.next_blocks.append(else_block.block_id)
                initial_branch.fall_through_block = if_block.block_id
                initial_branch.branching_block = else_block.block_id

                # set current_block
                self.curBlock_id = join_block.block_id

            else:
                raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['statSequence']}\"")
        else:  # "if" relation "then" statSequence
            # when there was no else: fall through block is the join block

            # set join_block on if-side branch
            # we use else_block as join_block
            if_side_last_block = self.tinyBlocks[self.get_last_block_id(if_block.block_id)]
            # if_side_last_block.next_blocks.append(else_block.block_id)
            if_side_last_block.branching_block = else_block.block_id

            # inherit varTable from last block, and create phi function if needed
            for varName, var_id in if_side_last_block.varTable.items():
                if else_block.getVar(varName) == -1 or else_block.getVar(varName) == var_id:  # no need for phi
                    else_block.varTable[varName] = var_id
                else:  # has a conflict: need a phi function
                    else_block.varTable[varName] = self.makeSSA(else_block.block_id, TokenKind.PHI, if_side_last_block.getVar(varName), initial_branch.getVar(varName))

            # set branching block's next block
            initial_branch.fall_through_block = if_block.block_id
            initial_branch.branching_block = else_block.block_id

            # if the block is empty, we add empty to guide through
            if not else_block.SSA_ids:
                self.makeSSA(else_block.block_id, TokenKind.EMPTY)
            self.SSAs[branching_SSA].right = else_block.SSA_ids[0]

            # bra statement for
            self.makeSSA(if_side_last_block.block_id, TokenKind.BRA, else_block.SSA_ids[0])

            # set current_block
            self.curBlock_id = else_block.block_id

        if self.check(TokenKind.FI):  # "if" relation "then" statSequence "fi"
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
            header_block = self.tinyBlocks[self.makeBlock()]      # header_block: aka join block
            header_block.varTable = deepcopy(initial_branch.varTable)
            header_block.opDomTree = deepcopy(initial_branch.opDomTree)
            self.curBlock_id = header_block.block_id

            branching_SSA = self.relation()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['relation']}\"")

        if self.check(TokenKind.DO):  # "while" relation "do"
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"do\"")

        if self.checkForTerminal("statSequence"):  # "while" relation "do" StatSequence
            # creating loop block and else block
            loop_block = self.tinyBlocks[self.makeBlock()]      # fall through
            else_block = self.tinyBlocks[self.makeBlock()]      # loop condition not satisfied: branching block

            # set initial_branch's next blocks:
            initial_branch.fall_through_block = header_block.block_id
            header_block.fall_through_block = loop_block.block_id
            header_block.branching_block = else_block.block_id

            # # add empty if initial branch is empty:
            # if not initial_branch.SSA_ids:
            #     self.makeSSA(initial_branch.block_id, TokenKind.EMPTY)

            # inheriting the variable Table and opDomTree
            loop_block.varTable = deepcopy(initial_branch.varTable)
            loop_block.opDomTree = deepcopy(initial_branch.opDomTree)

            # setting the branching statement
            self.makeSSA(else_block.block_id, TokenKind.EMPTY)
            self.SSAs[branching_SSA].right = else_block.SSA_ids[0]

            # making contents within loop_block
            self.curBlock_id = loop_block.block_id
            self.statSequence()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting either of \"{self.first['statSequence']}\"")

        if self.check(TokenKind.OD):  # "while" relation "do" StatSequence "od"
            # looping back to beginning
            loop_side_last_block = self.tinyBlocks[self.get_last_block_id(loop_block.block_id)]
            loop_side_last_block.branching_block = header_block.block_id
            loop_side_last_block.join_block = header_block.block_id
            # add empty if initial branch is empty:
            if not header_block.SSA_ids:
                self.makeSSA(header_block.block_id, TokenKind.EMPTY)
            self.makeSSA(loop_side_last_block.block_id, TokenKind.BRA, header_block.SSA_ids[0])   # headerbranch guaranteed to have SSA because we added empty earlier

            # header-commit:
            for ssa_id in header_block.SSA_ids:
                ssa = self.SSAs[ssa_id]
                if ssa.op == TokenKind.PHI:
                    name = header_block.getName(ssa_id)
                    after = ssa_id
                    self.loop_update(header_block.block_id, name, after)

            # inherit
            else_block.varTable = deepcopy(header_block.varTable)
            else_block.opDomTree = deepcopy(header_block.opDomTree)
            self.curBlock_id = else_block.block_id
            self.next()
        else:
            raise SyntaxError(f"index:{self.tokenizer.index} - Invalid use of {self.cur} when expecting \"od\"")

    def loop_update(self, block_id:int, varName: str, after:int):
        def phi_propagation(update_blocks:list[int], block_id: int, name:str, after: int):
            # check if we already updated this block
            # terminating condition
            if (block_id is None) or (update_blocks[block_id] == 1):
                return

            # search for all instances of before, swap it to after
            block = self.tinyBlocks[block_id]

            # varTable update
            before = block.getVar(name)
            block.assignVar(name, after)

            # SSA sentence update
            for _ssa_id in block.SSA_ids:
                _ssa = self.SSAs[_ssa_id]
                if _ssa.left == before:
                    _ssa.left = after
                if _ssa.right == before:
                    _ssa.right = after
                # if ssa.op == TokenKind.PHI:
                #     return

            # mark updated
            update_blocks[block_id] = 1

            # further propagate
            phi_propagation(update_blocks, block.fall_through_block, name, after)
            phi_propagation(update_blocks, block.branching_block, name, after)

        # ---------------------------------------------------------------- #

        # initial setup variables
        cur_block = self.tinyBlocks[block_id]
        before = cur_block.getVar(varName)

        # SSA sentence update
        for ssa_id in cur_block.SSA_ids:
            ssa = self.SSAs[ssa_id]
            if after != ssa_id:
                if ssa.left == before:
                    ssa.left = after
                if ssa.right == before:
                    ssa.right = after

        # propagate init
        update_blocks_ = [0 for _ in range(len(self.tinyBlocks))]      # marks
        update_blocks_[cur_block.block_id] = 1
        phi_propagation(update_blocks_, cur_block.fall_through_block, varName, after)
        phi_propagation(update_blocks_, cur_block.branching_block, varName, after)



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

