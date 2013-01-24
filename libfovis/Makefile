.DEFAULT_GOAL=all

GIT_DIR=fovis
GIT_URL=https://code.google.com/p/fovis/
GIT_REVISION=1098ed556d29f947591bcde3d6c4868d56c12bf5

include $(shell rospack find mk)/git_checkout.mk

all: fovis
	make -C fovis/libfovis
