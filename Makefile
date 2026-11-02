CXX ?= c++
VERSION ?= 0.1.0
SDK ?= ../lmcode-cpp
SDK_REF ?= HEAD
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -pedantic
INCLUDES := -Isrc -Ibuild -Ivendor/lmcode/include -Ivendor/lmcode/stdlib -DLMSEARCH_VERSION=\"$(VERSION)\"
BIN ?= build/lmsearch
LDFLAGS ?= $(if $(filter Windows_NT,$(OS)),-lws2_32,)
SCHEMA := schema/lmsearch.dsl.json
CLI_SCHEMA := cli.schema.latest.json
SCHEMA_HEADER := build/lmsearch_schema.hpp
SOURCES := src/main.cpp src/engine.cpp src/bindings.cpp
HEADERS := src/lmsearch.hpp vendor/lmcode/include/dslvm.hpp $(wildcard vendor/lmcode/stdlib/*.hpp)

# Optional upstream static libraries built in third_party/{faiss,onnx}
FAISS_LIB := third_party/faiss/build/faiss/libfaiss.a
ONNX_LIB := third_party/onnx/build/libonnx.a
ONNX_PROTO_LIB := third_party/onnx/build/libonnx_proto.a
UPSTREAM_INCLUDES := -Ithird_party/faiss -Ithird_party/onnx -Ithird_party/onnx/build -DONNX_ML=1 -DONNX_NAMESPACE=onnx -DLMSEARCH_WITH_UPSTREAM_FAISS_ONNX=1
UPSTREAM_LIBS := $(FAISS_LIB) $(ONNX_LIB) $(ONNX_PROTO_LIB) -lprotobuf -lopenblas -fopenmp -lpthread

.PHONY: all with-upstream test sync-sdk check-sdk clean

all: $(BIN)

$(SCHEMA_HEADER): $(SCHEMA) $(CLI_SCHEMA) scripts/embed-schema.py
	python3 scripts/embed-schema.py $(SCHEMA) $(SCHEMA_HEADER) $(CLI_SCHEMA)

$(BIN): $(SOURCES) $(HEADERS) $(SCHEMA_HEADER)
	@mkdir -p $(dir $(BIN))
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(LDFLAGS) -o $(BIN)

with-upstream: $(SOURCES) $(HEADERS) $(SCHEMA_HEADER) $(FAISS_LIB) $(ONNX_LIB)
	@mkdir -p build
	$(CXX) -std=c++20 -O2 -Wall -Wextra -fopenmp $(INCLUDES) $(UPSTREAM_INCLUDES) $(SOURCES) $(UPSTREAM_LIBS) -o build/lmsearch-upstream

sync-sdk:
	python3 scripts/sync.py sdk "$(SDK)" "$(SDK_REF)"

check-sdk:
	python3 scripts/sync.py check

test: $(BIN)
	bash ./test.sh

clean:
	rm -rf build dist
