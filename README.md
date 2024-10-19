# CompilerFromScratch
 Compiles tiny programming language into ir diagram (dot language)

 # Example
 Below is the sample code of fibonaci sequence in *tiny* programming language
 
 ```
 main
 var x;
  function fibonacci(n); {
    if n <= 1 then
      return n
    fi;
    return call fibonacci(n - 1) + call fibonacci(n - 2)
  };
  {
    let x <- call InputNum;
    let x <- call fibonacci(x);
    call OutputNum(x);
    call OutputNewLine
  }.
```

Below is the visualization of the IR representation produced by the project

![image](https://github.com/user-attachments/assets/e3d77e98-963f-42fb-adaf-eca398fcfe62)
