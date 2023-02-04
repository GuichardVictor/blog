---
title: "Rust Triton Client"
date: 2023-02-03T16:24:50+01:00
draft: false
tags:
  - ml
  - rust
categories:
  - programming
---

I've recently tried to improve model deployement and test different approaches.
After trying `Tensorflow Serving` and `Torch Serve`, I decided to take a look
at `Nvidia Triton`. Its high performance and multiple model backends is very
appealing. However I wanted to integrate it to my rust backend stack. Therefore,
I decided to implement a rust version of the GRPC Client.

<!--more-->


## Triton Client

`Proto` files can compiled into different programming languages.
In `Rust`, [`prost`](https://github.com/tokio-rs/prost) can be used to generate simple `Rust` code from `proto` files.
Which can also be used with [`tonic`](https://github.com/tokio-rs/prost) to write production ready code that uses `gRPC`.

### Retreiving the Triton Server Protos

After creating a new rust project, we can retreive the `proto` files defined by nvidia using git submodules:

```sh
git submodule add git@github.com:triton-inference-server/common.git
```

The `protos` are defined in `/common/protobuf/`.

The advantages of using a submodule is if the code is updated by nvidia we will spend less time to update our dependencies.

### Generating Rust Code

To generate `rust` code from these `protos` we will need to add these dependencies:

```sh
cargo add prost
cargo add tonic
cargo add --build tonic-build
```

We then need to write some code in the `/build.rs` file:

```rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pb_dir: std::path::PathBuf = env::var("TRITON_PROTOBUF")
        .ok()
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/common/protobuf").to_string())
        .into();

    let protobuf_paths: Vec<_> = ["grpc_service.proto", "health.proto", "model_config.proto"]
        .map(|protoname| pb_dir.join(protoname))
        .to_vec();

    tonic_build::configure()
        .build_server(true)
        .compile(&protobuf_paths, &[&pb_dir])?;

    Ok(())
}
```

This part will read from the environement the source directory of the triton protos and return a path:

```rs
let pb_dir: std::path::PathBuf = env::var("TRITON_PROTOBUF")
    .ok()
    .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/common/protobuf").to_string())
    .into();
```

We then need to get the complete path of each protos we want to generate rust code:

```rs
let protobuf_paths: Vec<_> = ["grpc_service.proto", "health.proto", "model_config.proto"]
    .map(|protoname| pb_dir.join(protoname))
    .to_vec();
```

And finally call `tonic_build` to generate the rust code the defined output directory:
```rs
tonic_build::configure()
    .build_server(true)
    .compile(&protobuf_paths, &[&pb_dir])?;
```

and call

```sh
cargo build
```

### Using the generated code

Now that we have generated the rust code from our `proto` files we can include it in our `lib.rs` as a mod:

```rs
pub mod triton {
    include!(concat!(env!("OUT_DIR"), "/inference.rs"));
}
```

And start sending gRPC messages to the Triton Inference Server with [`tokio-rs`](https://github.com/tokio-rs/tokio):

```rs
let url = env::var("TRITON_HOST").ok().unwrap_or("http://localhost:8001");
let mut client = GrpcInferenceServiceClient::connect(url.into()).await.unwrap();
let response = client.server_live(ServerLiveRequest {}).await.unwrap();
println!("{:?}", response.into_inner()) // OK => the server is live :D
```

### Improvements

We can improve the current code by implementing wrappers on the `GrpcInferenceServiceClient` and the different messages such as builders.
