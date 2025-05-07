f2py -h ttplanet.pyf -m ttplanet ttplanet.f --overwrite-signature
f2py -c ttplanet.pyf -m ttplanet ttplanet.f \
  --verbose \
  --backend='meson'\
  --f77flags="-std=legacy" \
