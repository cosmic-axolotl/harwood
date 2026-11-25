#!/bin/sh
# RVSAO 2.8.5 with emplot's two calls to fillpix, and plotspec's, given their fifth
# argument, for the cases that delete part of the spectrum (FIXED in ep_cases.py
# and ../iraf_xcsao/xp_cases.py):
#
#     sh build_fixed.sh <rvsao folder> <folder>  ->  <folder>/rvsao/bin/x_rvsao.e
#
# emplot.x calls fillpix (npoints, pix, ipx1, ipx2) -- plotspec.x fillpix (npoints,
# pix, d1, d2) -- and fillpix (Util/fill.x) is
# fillpix (npts, pix, ip1, ip2, debug): its `if (debug)` reads through an address
# never passed, and the d key's second press ends emsao in a segmentation violation,
# always. With the argument passed, d does what its code says; harwood does that.
# <rvsao folder> is RVSAO's, built (build.sh); PATH is the system's, as there.
set -e
src=$(cd "$1" && pwd)
dest=$(mkdir -p "$2" && cd "$2" && pwd)
export PATH=/usr/bin:/bin
rm -rf "$dest/rvsao"
cp -r "$src" "$dest/rvsao"
cd "$dest/rvsao"
sed -i 's/call fillpix (npoints, specwcs, ipx1, ipx2)/call fillpix (npoints, specwcs, ipx1, ipx2, debug)/
s/call fillpix (npoints, pix, ipx1, ipx2)/call fillpix (npoints, pix, ipx1, ipx2, debug)/' Emsao/emplot.x
test "$(grep -c 'ipx2, debug)' Emsao/emplot.x)" = 2
sed -i 's/call fillpix (npoints, pix, d1, d2)/call fillpix (npoints, pix, d1, d2, debug)/' \
    Util/plotspec.x
test "$(grep -c 'd2, debug)' Util/plotspec.x)" = 1
sleep 1
touch Emsao/emplot.x Util/plotspec.x
rm -f bin && ln -s bin.linux64 bin
export iraf=/usr/lib/iraf/ IRAFARCH=linux64 noao=/usr/lib/iraf/noao/ rvsao="$dest/rvsao/"
mkpkg -p noao nrelink > build.log 2>&1 || { tail -20 build.log; exit 1; }
grep -q "emplot.x" build.log
grep -q "plotspec.x" build.log
mv xx_rvsao.e bin.linux64/x_rvsao.e
ls "$dest/rvsao/bin/x_rvsao.e"
