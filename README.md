# Problems

Symmetric TSP problems, but in the XML structure each cost appreas two times so the algorithm can be designed for non-symmetric.

The goal is to find the round trip that has minimal cost.

# loading a xml in Python
```
from lxml import etree

with open(filename) as f:
    xml = etree.fromstring(f.read())    # root element of xml
# get all vertices
vertices = xml.find('graph').findall('vertex')
# get the edges of first vertex
edges = vertices[0].findall('edge')
# get cost and target vertex of first edge
cost = edges[0].get('cost')
target_vertex = edges[0].text

```

