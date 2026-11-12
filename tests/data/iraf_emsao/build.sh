#!/bin/sh
# Builds RVSAO 2.8.5, the gabarito of tests/test_emsao_rvsao.py, against the
# installed IRAF 2.18 (the Debian package, /usr/lib/iraf):
#
#     sh build.sh <folder>      ->  <folder>/rvsao-2.8.5/bin/x_rvsao.e
#
# RVSAO is the Smithsonian Astrophysical Observatory's radial velocity package
# (http://tdc-www.harvard.edu/iraf/rvsao/), an IRAF external package. PATH is the
# system's: a toolchain earlier in PATH (a MESA SDK's gcc and ld) links
# executables that fall over at IRAF's startup.
set -e
dest=$(mkdir -p "$1" && cd "$1" && pwd)
export PATH=/usr/bin:/bin
if [ ! -f "$dest/rvsao-2.8.5.tar.gz" ]; then
    curl -sL -o "$dest/rvsao-2.8.5.tar.gz" http://tdc-www.harvard.edu/iraf/rvsao/rvsao-2.8.5.tar.gz
fi
rm -rf "$dest/rvsao-2.8.5" "$dest/unpack"
mkdir "$dest/unpack"
tar xzf "$dest/rvsao-2.8.5.tar.gz" -C "$dest/unpack"
mv "$dest/unpack/rvsao-2.8.5" "$dest/rvsao-2.8.5"
rmdir "$dest/unpack"
cd "$dest/rvsao-2.8.5"
rm -f bin && ln -s bin.linux64 bin
export iraf=/usr/lib/iraf/ IRAFARCH=linux64 noao=/usr/lib/iraf/noao/ rvsao="$dest/rvsao-2.8.5/"
mkpkg -p noao nrelink > build.log 2>&1 || { tail -20 build.log; exit 1; }
mv xx_rvsao.e bin.linux64/x_rvsao.e
ls "$dest/rvsao-2.8.5/bin/x_rvsao.e"
