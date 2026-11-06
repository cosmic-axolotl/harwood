#!/bin/sh
# Builds IRAF 2.18.1's odcombine, the gabarito of tests/test_odcombine_iraf.py.
#
#     sh build.sh <folder>      ->  <folder>/odc/xx_odcombine.e and odcombine.par
#
# The odcombine installed with IRAF 2.18.1 never writes an output: it lists the
# spectra it split with the template "%s.*\\[^x]" (t_odcombine.x:275), which the
# imt of 2.18 matches to nothing, and says "No images to combine". The same line is
# still in IRAF's main. odcombine-2.18.1.patch changes only that template, and its
# bad-pixel twin, to "%s.*" (the cases are FITS; [^x] kept out imh's .pix files, and
# "%s.[0-9]*" will not do: imt reads [...] as an image section), and corrects
# x_odcombine.x, which declares `task scombine = t_scombine`.
#
# PATH is the system's: a toolchain earlier in PATH (a MESA SDK's gcc and ld) links
# executables that fall over at IRAF's startup.
set -e
here=$(cd "$(dirname "$0")" && pwd)
dest=$(mkdir -p "$1" && cd "$1" && pwd)
export PATH=/usr/bin:/bin
if [ ! -d "$dest/iraf-src" ]; then
    git -c advice.detachedHead=false clone -q --depth 1 --branch v2.18.1 --filter=blob:none --sparse \
        https://github.com/iraf-community/iraf.git "$dest/iraf-src"
    git -C "$dest/iraf-src" sparse-checkout set noao/onedspec/odcombine
fi
rm -rf "$dest/odc"
cp -r "$dest/iraf-src/noao/onedspec/odcombine" "$dest/odc"
patch -s -d "$dest/odc" -p1 < "$here/odcombine-2.18.1.patch"
cd "$dest/odc"
export iraf=/usr/lib/iraf/ IRAFARCH=linux64 noao=/usr/lib/iraf/noao/
mkpkg -p noao standalone > build.log 2>&1 || { tail -20 build.log; exit 1; }
ls "$dest/odc/xx_odcombine.e"
