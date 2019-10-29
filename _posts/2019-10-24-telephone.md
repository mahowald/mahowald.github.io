---
layout: post
title: "A minimal RESTful API wrapper"
tags: [docker, golang]
author: Matthew Mahowald
mathjax: true
---

These days, just about everything in the cloud gets deployed as a containerized microservice with a RESTful API.
Most languages have some capability for spinning up microservers, either built-in or facilitated by third-party packages (such as Python's [FastAPI](https://github.com/tiangolo/fastapi) and R's [Plumber](https://www.rplumber.io/).
But what if you're completely uninterested in delivering your code as a microservice with a REST interface?
What if in fact you'd rather just write code that doesn't even know it's running in some server?
Then, dear reader, this blog post is for you.

The setup
---------

Let's start with a simple Python script:

```python
import sys

for line in sys.stdin:
    print(int(line) + 1)
    sys.stdout.flush()
```

This script reads from stdin, attempts to cast each line to an integer, add one to that integer, and write the output to stdout.
This is obviously a contrived example, but it fits into a general pattern:

* Input is received over stdin and output put onto stdout.
* The script is _asynchronous_ and _streaming_, i.e. inputs are read in one at a time, processed, and outputs are written to an output pipe (stdout).

Note that these characteristics are shared with most of the unix tools you're used to working with (`grep`, `cat`, etc). In particular, these characteristics make this script amenable to piping, e.g.
```
cat myfile.txt | python test.py >> outputs.txt
```

A Golang wrapper
----------------

In [a previous post]({{ site.baseurl}}{% link _posts/2019-6-13-go-ffi.md %}), I wrote about using foreign function interfaces to share data between languages.
Our approach here will be much simpler.
We're going to write a small program in Go that will kick off our script as a subprocess, and pass data in and out using stdin and stdout.
To make a simple command line interface, I'll use the excellent [`cli` package from urfave](https://github.com/urfave/cli).
Our Go program will be short enough to fit into just one `main.go` file.
Here's the skeleton:

```go
package main

import (
    "os/exec"
    "github.com/urfave/cli"
    "log"
)

func main() {
    app := cli.NewApp()
    app.Name = "telephone"

    app.Action = func(c *cli.Context) error {
        cmd := exec.Command(c.Args().First(), c.Args().Tail()...)

        cmdWriter, _ := cmd.StdinPipe()
        cmdReader, _ := cmd.StdoutPipe()

        err := cmd.Start()
        if err != nil {
            log.Fatal(err)
        }
        
        err = cmd.Wait()
        return err
    }

    err := app.Run(os.Args)
    if err != nil {
        log.Fatal(err)
    }
}
```

As written, this code is enough to kick off the command of our choice and connect up pipes to its stdin and stdout.
Note that I've used `cmd.StdinPipe()` and `cmd.StdoutPipe()`, rather than using `cmd.Stdin` and `cmd.Stdout`.
`cmd.Stdin` and `cmd.Stdout` are just byte buffers, whereas the corresponding `*Pipe()` functions implement `io.WriteCloser` and `io.ReadCloser` interfaces, making it easy for us to pass data back and forth between the subprocess and our Go program.
We need to set up some functions to actually write to the stdin pipe and read from the stdout pipe, however.
A clean way to do this is to set up some channels and then kick off goroutines to handle passing data in and out of the subprocess via stdin/stdout:

```go
inchan := make(chan string)
outchan := make(chan string)

// read from stdout
go func() {
    scanner := bufio.NewScanner(cmdReader)
    for scanner.Scan() {
        msg := scanner.Text()
        outchan <- msg
    }
}()

// write to stdin
go func() {
    for msg := range inchan {
        io.WriteString(cmdWriter, msg+"\n")
    }
}()
```

Now, passing data in to the script is as easy as pushing a string onto `inchan`, and we can read the subprocess's stdout by pulling from `outchan`.

Adding a microserver
--------------------

The remaining piece of the puzzle is adding a small RESTful server to allow us to supply inputs and return responses over HTTP.
Go makes starting an HTTP server extremely easy; the general syntax is just

```go
http.HandleFunc(<PATH>, handler)

log.Fatal(http.ListenAndServe(":PORT", nil))
```

where `handler` is the function we want to bind to `<PATH>`, and `PORT` is the port we want to use for our server.
The only slightly nontrivial part of this is the `handler` function; this has to be a function that takes an `http.ResponseWriter` and a pointer to an `http.Request` as arguments, and returns nothing.
How do we wire in the channels (`inchan` and `outchan`) that we set up earlier?

Go is actually a surprisingly functorial language, so one way to do this is to define a function that returns a function with the appropriate signature. Take a look:

```go
func makeRequestHandler(
    inputs chan string, 
    outputs chan string,
) func(w http.ResponseWriter, r *http.Request) {
    f := func(w http.ResponseWriter, r *http.Request) {
        switch r.Method {
        case "POST":
            body, err := ioutil.ReadAll(r.Body)
            if err != nil {
                http.Error(w, err.Error(), 500)
            }
            inputs <- string(body)
            resp := <-outputs
            fmt.Fprint(w, resp+"\n")
        }
    }

    return f
}
```

Putting it all together, here's our updated `main()` function:

```go
func main() {
    app := cli.NewApp()
    app.Name = "telephone"
    app.Usage = "Wrap the specified application with a simple webserver"

    app.Flags = []cli.Flag{
        cli.StringFlag{
            Name:  "port",
            Value: "8080",
            Usage: "port for the webserver",
        },
    }

    app.Action = func(c *cli.Context) error {
        cmd := exec.Command(c.Args().First(), c.Args().Tail()...)
        
        cmdWriter, _ := cmd.StdinPipe()
        cmdReader, _ := cmd.StdoutPipe()
        
        inchan := make(chan string)
        outchan := make(chan string)
        
        go func() {
            scanner := bufio.NewScanner(cmdReader)
            for scanner.Scan() {
                msg := scanner.Text()
                outchan <- msg
            }
        }()
        
        go func() {
            for msg := range inchan {
                io.WriteString(cmdWriter, msg+"\n")
            }
        }()

        err := cmd.Start()
        if err != nil {
            panic(err)
        }

        http.HandleFunc("/", makeRequestHandler(inchan, outchan))
        go func() {
            log.Fatal(http.ListenAndServe(fmt.Sprintf(":%v", c.String("port")), nil))
        }()

        err = cmd.Wait()
        return err
    }

    err := app.Run(os.Args)
    if err != nil {
        log.Fatal(err)
    }
}
```

And that's it!

You can check out a complete copy of this miniproject, including a Dockerfile for containerization, [on my GitHub](https://github.com/mahowald/telephone).