.PHONY: clean build prep all data-fundamentals python stats linear-algebra pandas

# Build all slides
build: prep data-fundamentals

# Build preparation slides
prep:
	@echo "Building preparation slides..."
	@node slides/build.js 0-prep

# Build data fundamentals slides
data-fundamentals: data-analytics python stats linear-algebra pandas

# Build individual module slides
data-analytics:
	@echo "Building data analytics slides..."
	@node slides/build.js 1-data-fundamentals/1.1-intro-data-analytics

python:
	@echo "Building Python slides..."
	@node slides/build.js 1-data-fundamentals/1.2-intro-python

stats:
	@echo "Building statistics slides..."
	@node slides/build.js 1-data-fundamentals/1.3-intro-statistics

linear-algebra:
	@echo "Building linear algebra slides..."
	@node slides/build.js 1-data-fundamentals/1.4-data-foundation-linear-algebra

pandas:
	@echo "Building pandas slides..."
	@node slides/build.js 1-data-fundamentals/1.5-data-analysis-pandas

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@find . -type f -name "index.html" -delete

# Clean specific module
clean-module:
	@echo "Cleaning specific module..."
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: MODULE variable not set. Usage: make clean-module MODULE=module-name"; \
		exit 1; \
	fi
	@find . -path "*/$(MODULE)/*" -type f -name "index.html" -delete

# Regenerate specific module
regenerate:
	@echo "Regenerating specific module..."
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: MODULE variable not set. Usage: make regenerate MODULE=module-name"; \
		exit 1; \
	fi
	@make clean-module MODULE=$(MODULE)
	@node slides/build.js 1-data-fundamentals/$(MODULE)

# Default target
all: build 
