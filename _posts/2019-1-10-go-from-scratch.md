---
layout: post
title: "Building containerized RESTful microservices from scratch"
tags: [docker, swagger, golang]
---

A huge part of my job at Open Data Group involves delivering
software and models as containerized microservices. 
Because we find ourselves
doing this a lot, my team has standardized around a
systematic approach using existing tooling to allow us to
rapidly prototype and develop production-grade containerized
applications. In this blog post, I'll talk about three parts
to this approach, specifically focusing on applications
implemented in [Go](https://golang.org/).

Why Go?
-------

In this post, I'm going to specifically focus on the most
common subset of containerized microservices: lightweight,
limited-functionality webservers implementing a RESTful
API. From [Python](http://flask.pocoo.org/) to
[Erlang](https://github.com/ninenines/cowboy), just about
every language has a great framework for easily building out
a server. Go, however, has a couple of features that make it
a particularly appealing choice of language for a
production-grade containerized microservice:

* A very strong standard library as well as a rich ecosystem
  of community packages
* Highly performant, simple to use concurrency abstractions
* The ability to easily generate statically linked binaries

The last point is especially valuable when targeting Docker
as the code deployment mechanism.

A Step-By-Step guide to building a RESTful microservice
-------------------------------------------------------

In contrast to some traditional development patterns,
when building out a RESTful microservice at Open Data Group,
we like to take an "outside in" approach. By this I mean that
we start by designing the REST API of the service we're
building, then generate a server stub, and finally start
filling out the "inside" (i.e. implementing the underlying
functionality). There's two reasons for this approach:

1. Most RESTful microservices do not exist in isolation:
   by beginning with an interface definition, development
   of and integration with other components can be done in parallel. And, because
   the server stub can be rapidly generated from that
   interface definition, we can naturally apply a test-driven
   development methodology to guide and check the ultimate
   implementation of the underlying functionality.
2. In general, interfaces tend to be longer-lived than
   the innards of the service: other services or client
   applications may depend on functionality defined in
   that particular interface, so by starting with the
   interface design we ensure that we're focusing our
   attention first on what is generally the most critical
   part of the application.

Additionally, we leverage a standard, language-neutral
format for defining our API specifications: [OpenAPI (formerly known as Swagger)](https://swagger.io/specification/).
OpenAPI is designed and guided by [Swagger](https://swagger.io/),
who also publishes a number of tools for working with these API
specs.
The OpenAPI
specification is sufficiently general to capture most of the
functionality we've ever been interested in implementing. In
my experience, it is particularly well-suited to describing
API calls where the request or response are encoded as JSON
objects; it can be a little more cumbersome for edge cases
where the request encoding doesn't map to JSON or YAML.

Here's an example OpenAPI specification for a very simple
microservice:

```yaml
swagger: "2.0"
info:
    description: "An example FastScore microservice."
    version: "0.1"
    title: "My FastScore Microservice"

paths:
    /:
        get:
            produces:
                - text/html
            operationId: get_index
            responses:
                200:
                    description: Returns the Dashboard homepage.
                    schema: { type: string }
    /{engine}/model:
        get:
            parameters:
                - name: engine
                  description: The name of the engine instance
                  type: string
                  in: path
                  required: true
            produces:
                - application/json
            operationId: get_model
            responses:
                200:
                    description: Returns a description of the currently running model.
                    schema: { type: object }
```

Generating a server stub
------------------------

Swagger provides a number of tools for working with OpenAPI
specifications, including one called [`swagger-codegen`](https://swagger.io/tools/swagger-codegen/), which
can take in an API specification and generate client and
server stubs. However, the official `swagger-codegen` has a few issues
with its Go support, so instead we use
[the Go package `go-swagger`](https://github.com/go-swagger/go-swagger).

To generate a server stub with `go-swagger`, just run
```
swagger generate server -f api.yaml
```
This will spit out a few different directories:

- `cmd`, whose subdirectory contains the main entrypoint
- `models`, which contains definitions for all of the types
  used in your API
- `restapi`, which contains the actual API implementation.

For convenience, let's assume that the name of our microservice
is `myservice`. Of these, at a minimum the two files we need to
edit are `main.go`, located in `cmd/myservice-server`, and
`configure_myservice.go`, located in `restapi`.

The latter of these files we need to edit to implement the actual
functionality of our service. This consists of filling out the various
API functions defined in this file (the functions named `api.NameOfOperation`)
and replacing the `middleware.NotImplemented(...)` return values to appropriate
returns for the outcomes of the functions. A typical "filled out" API function
will look something like:

```go
api.NameOfOperation = operations.NameOfOperationFunc(func(params operations.NameOfOperationParams) middleware.Responder {

    result := DoSomething(params.Parameter1, params.Parameter2)

    return operations.NewNameOfOperationOK().WithPayload(result)
})
```

where `NameOfOperation` is the name of the operation defined in the OpenAPI spec,
`Parameter1` and `Parameter2` are two of the defined inputs for that function,
`result` is an appropriate response for this API call, and `DoSomething` is
the actual "under the hood" function that should be called.

Once this is done, you can build and run your server by navigating to the
`cmd/myservice-server` directory and running `go install`, and then running
(for example)
```
./myservice-server --tls-host=0.0.0.0 --tls-port=8080 --tls-key=keys/key.key --tls-certificate=keys/crt.crt
```
If all goes well, you'll see a log message that you're now serving
your API at port 8080 on your local machine, and you should be able to
connect to it from other machines.

Building a container *from scratch*
-----------------------------------

The astute reader will notice that I haven't described what should be modified
in `main.go`, nor have I taken advantage of Go's functionality around statically
linked binaries. Here's where both of these parts come in. But first, a slight
digression about Docker containers.

Most Docker images use a particular Linux distribution as their base. Among the
most common choices are Ubuntu and Alpine (Ubuntu because it's easy to add other
dependencies in subsequent layers, and Alpine because it's much smaller than
most other Linux flavors). So, if you're interested in getting the smallest
container image possible, your Dockerfile should start with `FROM alpine`, right?

Wrong! We can do better. There's a special, completely empty base layer we can
use in Docker called `scratch`. The advantage of using `scratch` as a base layer
is that we have complete control over what goes into the container image, which
is especially important for minimizing image sizes and satisfying security
requirements. Note that `scratch` behaves somewhat differently
from other base layers:

* There's no tty or terminal or anything that a user can attach to, so you can't
  get "into" the container with `docker exec`
* Key filesystem abstractions are missing, so you generally can't provide
  command-line arguments to applications you start in the container
* And [there are other differences](https://docs.docker.com/develop/develop-images/baseimages/).

One important thing that `scratch` base layers do still allow is setting
environment variables. With this in mind, there are two steps remaining before
we have our containerized microservice:

1. Change `main.go` to take arguments via environment variables.
2. Compile a statically linked binary of our server and put it in a
   Dockerfile based on scratch.

The first part is straightforward: just add a few lines to `main.go` to set
the various arguments the microservice needs. For example, to set the TLS
host via environment variable, after the line
```go
server.ConfigureAPI()
```
in `main.go`, add the line
```go
host, exists := os.LookupEnv("HOST")
if exists {
    server.TLSHost = host
} else {
    server.TLSHost = "0.0.0.0" // or whatever the default value should be
}
```

Putting it all together
-----------------------

Finally, we have to build the statically linked binary and put it
in a Dockerfile. A good practice for repeatable builds is to use
a [multi-stage build](https://docs.docker.com/develop/develop-images/multistage-build/). This essentially creates an intermediate Docker container
which is used to compile any relevant binaries, and then those assets
are copied over to the final output container. This gives a convenient
and transferable encapsulation of the build environment.

Here's an example of our typical Dockerfile for building a statically
linked Go server:

```
FROM golang:1.10.3-alpine3.8

RUN apk add git

COPY . /go/src/github.com/myorg/myservice
WORKDIR /go/src/github.com/myorg/myservice/restapi/

WORKDIR /go/src/github.com/myorg/myservice/cmd/myservice-server

RUN go get
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix nocgo

FROM scratch

COPY --from=0 /go/src/github.com/myorg/myservice/cmd/myservice-server/myservice-server /myservice-server
COPY keys /keys

ENV PORT 8123
ENV TLS_CERT keys/dummy.crt
ENV TLS_KEY keys/dummy.key
ENV TLS_HOST 0.0.0.0

CMD [ "/myservice-server" ]
```

To build the image, just run
```
docker build -t myorg/myservice -f Dockerfile
```
and then we're done!

Concluding thoughts
-------------------

By combining a standardized API specification, stub generation tools, and Go,
it's really easy and quick to build out lightweight, secure containerized
microservices. This has allowed my team to very quickly generate,
iterate on, and refine prototypes for new microservices, and keeps us close
to the guiding Docker philosophy of light, ephemeral, and portable application
design. Some additional related topics that I didn't cover here (but might in
a future post) include how to handle stateful containerized microservices,
equivalent practices for other languages such as Python, and building out
client libraries from an API specification. If you want to learn more
about these topics, [Swagger](https://swagger.io) and [Docker](https://www.docker.com/) both have a wealth of resources available. Happy coding!

_A version of this post originally appeared on [Open Data Group's tech blog](https://opendatagroup.github.io/development/2019/01/10/go-from-scratch.html)._