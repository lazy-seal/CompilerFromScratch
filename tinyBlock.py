from tinyParser import TokenKind
# from collections import defaultdict


"""
Should've started this project with C or C++...
"""

# class DomTree:
#     def __init__(self, _SSA_id=None, _prev=None, _next=None):
#         self.SSA_id = _SSA_id    # this is mainly an index within the parser
#         self.prev = _prev
#         self.next = _next        # do we need this?

class SSA:
    def __init__(self, _id: int,
                 op: TokenKind,
                 left = None,
                 right = None,
                 leftBlock = None,
                 rightBlock = None,
                 name = None,
                 paramNum = None):
        self.id = _id                   # id for the SSA: should be used to access SSA
        self.op = op                    # TokenKind Enum to represent operation: add, sub, mul, const, function_call, etc
        self.left = left                # id (int) of left SSA
        self.right = right              # id (int) of right SSA
        self.leftBlock = leftBlock      # this is for the phi, to indicate where the left SSA came from
        self.rightBlock = rightBlock    # same as above, but for right
        self.name = name                # name of the variable if it's the identifier
        self.paramNum = paramNum        # # of the param if op is the getpar or setpar

    def __repr__(self):
        _id = str(self.id)
        op = str(self.op)[10:]
        left = str(self.left)
        right = str(self.right)
        paramNum = str(self.paramNum)
        toReturn = ""

        if self.op == TokenKind.PHI:
            toReturn += f"({self.name}) "

        toReturn += _id

        if self.op == TokenKind.EMPTY:
            toReturn += ": \<empty\>"
            return toReturn

        toReturn += ": " + op

        if self.op == TokenKind.CONSTANT:
            toReturn += " #" + left
            return toReturn

        if self.op in [TokenKind.SETPAR, TokenKind.GETPAR]:
            toReturn += paramNum

        if self.left is not None:
            toReturn += f" ({left})"
            if self.right is not None:
                toReturn += f" ({right})"

        return toReturn

class tinyBlock:
    def __init__(self, _id: int):
        self.block_id = _id                         # this is mainly an index within the parser
        # self.opDomTree = defaultdict(DomTree)       # {op : linked_list} dictionary of linked list of Domination Tree per each operation
        self.varTable = {}                          # {name : SSA_id}
        self.usageTable = {}                        # {name : list[SSA_id]}
        self.SSA_ids = []                           # list of SSA_ids in order (of generation)
        # self.next_blocks = []
        self.fall_through_block = None              # next fall_through block
        self.branching_block = None                 # next branching block
        # self.join_block = None                      # join block
        self.prev = []                              # previous block
        # self.depth = None                         # depth level
        self.is_header_block = False                # indicates if the block is while loop header or not
        self.is_if_block = False
        self.dominated_by = None                    # the block that dominates this block
        self.dominates = []                         # block(s) that this block dominates
        self.function_calls = set()                 # functions(in block_id) that this block calls (visualization purpose)

    # def returnDominatingSSA(self, op: TokenKind, SSA_id: int) -> int:
    #     """
    #     :param op:
    #     :param SSA_id:
    #     :return: SSA_id of the dominating SSA. returns original SSA_id if don't have one
    #     """
    #     tree = self.opDomTree[op]
    #     SSA_id = tree.SSA_id
    #     # while tree:
    #     #     SSA_id = tree.SSA_id
    #     #     tree = tree.prev
    #     return SSA_id

    # def insertDomTreeAtTail(self, op: TokenKind, SSA_id: int) -> None:
    #     head = self.opDomTree[op]
    #
    #     if head.SSA_id is None:     # first initialization of DomTree
    #         head.SSA_id = SSA_id
    #         return
    #
    #     while head.next:    # at least 1 node in the thing
    #         head = head.next
    #     head.next = DomTree(SSA_id, _prev = head)

    def getName(self, ssa_id: int) -> str | None:
        for name, ssa in self.varTable.items():
            if ssa == ssa_id:
                return name
        return None

    def getVar(self, name)->int:
        """
        :param name: the name of the variable
        :return: SSA_id of the variable within this block's table
        :return: -1 means error (not defined)
        """
        if name is None:
            raise SyntaxError(f"Error occurred while getting the SSA_id: the variable name is None")
        try:
            to_return = self.varTable[name]
            return to_return
        except KeyError:
            return -1

    def assignVar(self, name, _id):
        """
        Initializes varTable[_name] to old _id
        Modifies varTable[name] to _id
        @TODO check for dominance tree and do a CSE (common subexpression elimination)
        :param name:
        :param _id:
        :return:
        """
        try:
            if type(name) is int:
                if self.block_id != 0:
                    raise SyntaxError("Error occurred while trying to store constant in nonzero (id) basic block")
                try:
                    self.varTable[name]
                except KeyError:
                    self.varTable[name] = _id
            elif type(name) is str:
                old_id = self.varTable[name]
                prev_id = old_id
                old_name = name + "_"
                while True:
                    try:
                        old_id = self.varTable[old_name]
                        old_name += "_"
                    except KeyError:
                        break
                # self.varTable[old_name] = prev_id
                self.varTable[name] = _id
            else:
                raise SyntaxError("Supposed to be a string or int")
        except KeyError:
            self.varTable[name] = _id



