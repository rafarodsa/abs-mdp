import re
import collections
import json 
import dreamerv3.embodied as embodied
from dreamerv3.embodied import path



class JSONLOutputRepeated(embodied.logger.JSONLOutput):
    
  def _make_lines(self, step, summary):
    lines = []
    for k, vs in summary.items():
        if len(vs) == 0:
          continue
        for i, v in enumerate(vs):
          if i > len(lines)-1:
            lines.append({'step': step})
          lines[i][k] = v
    return map(json.dumps, lines)

      
  def _write(self, summaries):
    bystep = collections.defaultdict(lambda: collections.defaultdict(list))
    for step, name, value in summaries:
      if len(value.shape) == 0 and self._pattern.search(name):
        bystep[step][name].append(float(value))
    lines = ''.join(['\n'.join(self._make_lines(step, scalars)) +'\n'
        for step, scalars in bystep.items()])
    print(lines)

    with (self._logdir / self._filename).open('a') as f:
      f.write(lines)