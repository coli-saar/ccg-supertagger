
import re
import json

regex = r"<L\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*>"
supertags = []

with open("../ccgbank_1_1/data/AUTO/00/wsj_0001.auto", "r") as f:
    for line in f:
        if line.startswith("("):
            supertags_here = []
            supertags.append(supertags_here)

            matches = re.finditer(regex, line, re.MULTILINE)
            for match in matches:
                supertags_here.append({"word": match.group(4), "supertags": [{"tag": match.group(1), "score": 1}]})
                # print(f"{match.group(1)}\t{match.group(4)}")

with open("supertags.json", "w") as out:
    json.dump(supertags, out)


