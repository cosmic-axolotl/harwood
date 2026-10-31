# IRAF 2.18 drawing fitsky's histplot and radplot: sh fitsky.sh NAME ALGORITHM SMOOTH CURSOR,
# from a directory with field.fits, field.coo ('40 46' and '111 52'), this folder's hcur
# and rcur, and home/.iraf/uparm, so that nothing is saved in the account's own. These need
# the image cursor, which PyRAF will only take from a display: the plain CL takes it as
# text from its input with stdimcur = text. Frames after the second are the rest of the
# list, gone through at the q. NAME.vdm is for ../iraf_photcal/decode_gki.py --everything.
name=$1; alg=$2; sm=$3; cur=$4
rm -rf home/.iraf/uparm/*
cat > "$name.cl" <<EOT
noao
digiphot
apphot
set stdimcur = "text"
datapars.fwhmpsf = 2.5
datapars.sigma = 6.
fitskypars.annulus = 10.
fitskypars.dannulus = 10.
fitskypars.salgorithm = "$alg"
fitskypars.smooth = $sm
fitskypars.khist = 3.
fitskypars.binsize = 0.1
fitsky field.fits coords=field.coo output=$name.sky interactive=yes icommands="" verify=no update=no verbose=no graphics=stdvdm display="" gcommands=$cur
40 46 1 \\040
111 52 1 \\040
0 0 1 q
yes
logout
EOT
HOME=$PWD/home timeout 30 irafcl < "$name.cl" > "$name.clout" 2>&1
mv home/.iraf/uparm/vdm "$name.vdm"
