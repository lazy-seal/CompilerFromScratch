from tinyParser import tinyParser
from tinyVisualizer import visualize
from tinyParser import TokenKind

def test_parser_funcCall():
    test1 = "main {call   \n func(a)}."
    test2 = "main {call   func(a, b, c, d, e)}."
    test3 = "main     {call func()}."
    test4 = "main {call func}."

    tester1 = tinyParser(test1)
    tester2 = tinyParser(test2)
    tester3 = tinyParser(test3)
    tester4 = tinyParser(test4)

    tester1.computation()
    tester2.computation()
    tester3.computation()
    tester4.computation()

    print("test_parser_funcCall Successful")

def test_parser_ifStatement():
    test1 = "main {if 3 > 2  \n   then let x <- a + b fi}."
    tester1 = tinyParser(test1)
    tester1.computation()

    test2 = "main {if 3 > 2 then   let x <- a + b else let y <- c + d fi}."
    tester2 = tinyParser(test2)
    tester2.computation()

    test3 = "main {if 3 - t * 22343 / ( 34342432 + x / 3 + tttsdfef) != asdf / 3 + 45 - tete then  \n let xdfs <- affff + be333f else let yasd <- cbdbdfbds + dwefwef fi}."
    tester3 = tinyParser(test3)
    tester3.computation()

    print("test_parser_ifStatement Successful")

def test_graph():
    FinalTest = "main \
var a, b, c, d; { \
    let a <- call InputNum(); \
    let b <- call InputNum(); \
    let c <- call InputNum(); \
    let d <- b; \
    while a < 100 do  \
        let a <- b + 1; \
        let d <- d + 1; \
        if d < 0 then  \
            let b <- b + 1; \
            while c < 10 do \
                let c <- c + 1; \
            od; \
            let d <- b + 1; \
        fi; \
    od; \
}."

    test1 = "main \
    var a, b; { \
        let a <- call InputNum(); \
        let b <- call InputNum(); \
        if a > 0 then \
            if b > 0 then \
                while a < 10 do \
                    let a <- a + 1;\
                od; \
            else \
                call OutputNum(1); \
            fi; \
        else \
            call OutputNum(b); \
            if b > 0 then \
                while a < 10 do \
                    let a <- a + 1;\
                od; \
            else \
                call OutputNum(1); \
            fi; \
        fi; \
        call OutputNum(a); \
    }."

    test2 = "main \
        var x, y, i, j; \
        { \
            let i <- call InputNum(); \
            let x <- 0; \
            let y <- 0; \
            let j <- i; \
            while x < 10 do \
                let x <- i + 1; \
                let y <- j + 1; \
                let j <- j + 1; \
            od; \
            od; \
            call OutputNewLine; \
    }."

    test3 = "main \
    var x, y, i, j; \
    { \
        let i <- call InputNum(); \
        let x <- 0; \
        let y <- 0; \
        let j <- i; \
        while x < 10 do \
            let x <- i + 1; \
            let y <- j + 1; \
            while j < 10 do \
                let x <- j + 1; \
                let y <- i + 1; \
                let j <- j + 1; \
            od; \
            let i <- i + 1 \
        od; \
        call OutputNum(x); \
}."

    three_layer_ifStatements = "main \
        var a, b; { \
            let a <- call InputNum(); \
            let b <- call InputNum(); \
            while a < 10 do \
                if a > 0 then \
                    if b > 0 then \
                        if b > 0 then \
                            let a <- a + 1; \
                        else \
                            let b <- 1 + b; \
                        fi; \
                    else \
                        if b > 0 then \
                            let a <- a + 1; \
                        else \
                            let b <- 1 + b; \
                        fi; \
                    fi; \
                else \
                    if b > 0 then \
                        if b > 0 then \
                            let a <- 1 + a; \
                        else \
                            let b <- b + 1; \
                        fi; \
                    else \
                        if b > 0 then \
                            let a <- 1 + a; \
                        else \
                            let b <- b + 1; \
                        fi; \
                    fi; \
                fi; \
            od; \
            call OutputNum(a); \
        }."
    tester1 = tinyParser(FinalTest)
    tester1.computation()
    visualize(tester1)
    tester1.printAllPhis()
    print()
    tester1.printAllVarTable()
    print()
    tester1.printAllSSAs()
    # print(tester1.variableModifiedLocation)
    # for block in tester1.tinyBlocks:
    #     print(f"block id: {block.block_id}")
    #     print(f"varTable: {block.varTable}")
    #     print()

def test_phi_location():
    tester = tinyParser("")

    # block making
    tester.makeBlock([])        # block 1
    tester.makeBlock([1])       # block 2
    tester.makeBlock([2])       # block 3
    tester.makeBlock([2])       # block 4
    tester.makeBlock([4])       # block 5
    tester.makeBlock([4])       # block 6
    tester.makeBlock([5])       # block 7
    tester.makeBlock([5, 6])    # block 8
    tester.makeBlock([7])       # block 9
    tester.makeBlock([9, 8])   # block 10

    # blocks:
    block1 = tester.tinyBlocks[1]
    block2 = tester.tinyBlocks[2]
    block3 = tester.tinyBlocks[3]
    block4 = tester.tinyBlocks[4]
    block5 = tester.tinyBlocks[5]
    block6 = tester.tinyBlocks[6]
    block7 = tester.tinyBlocks[7]
    block8 = tester.tinyBlocks[8]
    block9 = tester.tinyBlocks[9]
    block10 = tester.tinyBlocks[10]

    # prev due to while looping condition
    block1.prev.append(4)
    block2.prev.append(3)
    block7.prev.append(9)

    # domination relationship
    block2.dominated_by = 1
    block3.dominated_by = 2
    block4.dominated_by = 2
    block5.dominated_by = 4
    block6.dominated_by = 4
    block7.dominated_by = 5
    block8.dominated_by = 4
    block9.dominated_by = 7
    block10.dominated_by = 4

    print(tester.locationToPlacePhi([3, 6]))

def test_user_defined_func():
    fibonacci = "main \
        var x; \
         \
        function fibonacci(n); { \
            if n <= 1 then \
                return n \
            fi; \
            return call fibonacci(n - 1) + call fibonacci(n - 2) \
        }; \
         \
        { \
            let x <- call InputNum; \
            let x <- call fibonacci(x); \
            call OutputNum(x); \
            call OutputNewLine \
        }."

    mandelbrot = "main \
var px, py, mval; \
 \
function mandelbrot(x,y); \
var iters, x2, go,x0,y0;  \
{ \
    let x0 <- x; \
    let y0 <- y; \
    let iters <- 0; \
    let go <- 1; \
    while go != 0 do \
        if x*x+y*y > 4*10000*10000 then \
            let go <- 0; \
        fi; \
        if iters >= 100 then \
            let go <- 0; \
        fi; \
        if go != 0 then \
            let x2 <- (x*x-y*y)/10000 + x0; \
            let y <- (2*x*y)/10000 + y0; \
            let x <- x2; \
            let iters <- iters+1; \
        fi; \
    od; \
 \
    return iters; \
}; \
 \
{ \
    let px <- 0; \
    let py <- 0; \
    while py < 200 do \
        let px <- 0; \
        while px < 200 do \
            let mval <- call mandelbrot( ((px-100)*4*10000)/200, ((py-100)*4*10000)/200); \
            if mval == 100 then \
                call OutputNum(8); \
            else \
                call OutputNum(1); \
            fi; \
            let px <- px + 1; \
        od; \
        let py <- py + 1; \
        call OutputNewLine(); \
    od; \
 \
}."

    project_demo = "main \
var a, b, c, d; {\
\
    let a <- call InputNum();\
\
    let b <- call InputNum();\
\
    let c <- call InputNum();\
\
    let d <- b;\
\
    while a < 100 do \
\
        let a <- b + 1;\
\
        let d <- d + 1;\
\
        if d < 0 then \
\
            let b <- b + 1; \
\
            while c < 10 do \
\
                let c <- c + 1; \
\
            od; \
\
            let d <- b + 3;\
\
        else\
\
            let b <- b + 1;\
\
            while c < 10 do\
\
                let c <- c + 1;\
\
            od;\
\
            let d <- b + 3;\
\
        fi;\
\
    od;\
\
}."
    # print(mandelbrot)
    tester1 = tinyParser(fibonacci)
    tester1.computation()
    visualize(tester1)
    tester1.printAllVarTable()
    print()
    tester1.printAllSSAs()

if __name__ == "__main__":
    # test_parser_funcCall()
    # test_parser_ifStatement()
    # test_graph()
    # test_phi_location()
    test_user_defined_func()

