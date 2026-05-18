# Quick reference (short aliases):
#   make dev          — local site (Jekyll serve)
#   make site         — Jekyll build only
#   make search       — build + Pagefind index (search in built site)
#   make gems         — bundle install (Ruby deps)
#   make slides       — 0-prep + module 1 slide decks (same as build)
#   make all-slides   — every submodule with slides/data.json
#   make reg MOD=…    — clean + rebuild one module’s slides (path under docs/)
.PHONY: clean build build-all-slides prep all data-fundamentals regenerate clean-module \
	jekyll-serve jekyll-build bundle-install jekyll-build-with-search _ruby33 \
	dev site search gems slides all-slides reg validate-mermaid

# --- Short aliases (easier to type) ---
dev: jekyll-serve
site: jekyll-build
search: jekyll-build-with-search
gems: bundle-install
slides: build
all-slides: build-all-slides

# Regenerate one module’s slides: make reg MOD=2-data-wrangling/2.1-sql
reg:
	@if [ -z "$(MOD)" ]; then \
		echo "Usage: make reg MOD=path/under/docs  (e.g. MOD=2-data-wrangling/2.1-sql)"; \
		exit 1; \
	fi
	@$(MAKE) regenerate MODULE=$(MOD)

# Homebrew ruby@3.3 (keg-only) — avoids using default Homebrew ruby 4.x with this Gemfile
RUBY_33_PREFIX := $(shell brew --prefix ruby@3.3 2>/dev/null)
RUBY_33 := $(RUBY_33_PREFIX)/bin/ruby
RUBY33_BIN := $(RUBY_33_PREFIX)/bin

# Module 1 slide decks: short Make target | path under docs/
M1_PAIRS := \
	data-analytics|1-data-fundamentals/1.1-intro-data-analytics \
	python|1-data-fundamentals/1.2-intro-python \
	stats|1-data-fundamentals/1.3-intro-statistics \
	linear-algebra|1-data-fundamentals/1.4-data-foundation-linear-algebra \
	pandas|1-data-fundamentals/1.5-data-analysis-pandas

M1_TARGETS := $(foreach p,$(M1_PAIRS),$(word 1,$(subst |, ,$(p))))

.PHONY: $(M1_TARGETS)

define _m1_slide_rule
$(word 1,$(subst |, ,$(1))):
	@echo " -> $(word 2,$(subst |, ,$(1)))"
	@node slides/build.js $(word 2,$(subst |, ,$(1)))
endef

$(foreach p,$(M1_PAIRS),$(eval $(call _m1_slide_rule,$(p))))

# Build slides for 0-prep + module 1 (data fundamentals) only
build: prep data-fundamentals

# Rebuild every submodule that has slides/data.json
build-all-slides:
	@echo "Building all slide decks..."
	@for f in $$(find . \( -path './_site/*' -o -path './.git/*' \) -prune -o -path '*/slides/data.json' -type f -print); do \
		moddir=$$(dirname $$(dirname "$$f")); \
		echo " -> $$moddir"; \
		node slides/build.js "$$moddir" || exit 1; \
	done

prep:
	@echo " -> 0-prep"
	@node slides/build.js 0-prep

data-fundamentals: $(M1_TARGETS)

clean:
	@echo "Cleaning generated slide index.html files..."
	@find . -path '*/slides/index.html' -type f -delete

clean-module:
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: MODULE variable not set. Usage: make clean-module MODULE=module-name"; \
		exit 1; \
	fi
	@find . -path "*/$(MODULE)/*" -type f -name "index.html" -delete

regenerate:
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: MODULE variable not set. Example: make regenerate MODULE=2-data-wrangling/2.1-sql"; \
		exit 1; \
	fi
	@$(MAKE) clean-module MODULE=$(MODULE)
	@node slides/build.js $(MODULE)

all: build

# --- Jekyll (same Ruby major as GitHub Pages / Gemfile.lock) ---

_ruby33:
	@if [ ! -x "$(RUBY_33)" ]; then \
		echo "Install Ruby 3.3: brew install ruby@3.3"; \
		exit 1; \
	fi

bundle-install: _ruby33
	PATH="$(RUBY33_BIN):$$PATH" bundle install

# Auto-include _config.local.yml when present (used for local-only overrides
# like the chatbot endpoint pointing at vercel dev).
# $(comma) trick lets us put a literal comma inside a Make function arg.
comma := ,
LOCAL_CONFIG := $(wildcard _config.local.yml)
CONFIG_FLAG := $(if $(LOCAL_CONFIG),--config _config.yml$(comma)_config.local.yml,)

jekyll-build: _ruby33
	@PATH="$(RUBY33_BIN):$$PATH" bundle exec jekyll build $(CONFIG_FLAG)

jekyll-serve: _ruby33
	@PATH="$(RUBY33_BIN):$$PATH" bundle exec jekyll serve --livereload $(CONFIG_FLAG)

jekyll-build-with-search: _ruby33
	@PATH="$(RUBY33_BIN):$$PATH" bundle exec jekyll build $(CONFIG_FLAG)
	pnpm exec pagefind --site _site --output-path _site/pagefind

# Validate all .mmd diagram files
validate-mermaid:
	@bash scripts/validate-mermaid.sh
