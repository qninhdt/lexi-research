# Root Makefile for tau-research

.PHONY: check lint test smoke clean

check:
	@$(MAKE) -f ops/Makefile check

lint:
	@$(MAKE) -f ops/Makefile lint

test:
	@$(MAKE) -f ops/Makefile test

smoke:
	@$(MAKE) -f ops/Makefile smoke

clean:
	@$(MAKE) -f ops/Makefile clean
