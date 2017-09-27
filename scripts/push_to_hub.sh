#!/bin/sh

docker images | grep ta2/nyu_ta2 | awk '{ print $2; }' | while read i; do if [ "$i" != "<none>" ]; then docker tag registry.datadrivendiscovery.org/ta2/nyu_ta2:$i remram/d3m_nyu_ta2_vistrails:$i; docker push remram/d3m_nyu_ta2_vistrails:$i; fi; done
