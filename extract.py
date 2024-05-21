
# python extract.py ../ccgbank_1_1/data/AUTO/00/wsj_0001.auto 


import re
import json
import sys

regex = r"<L\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*>"
supertags = []

# print(sys.argv)
# sys.exit(0)

skipped_supertags = set(".,;:")

for filename in sys.argv[1:]:
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("("):
                supertags_here = []
                supertags.append(supertags_here)

                matches = re.finditer(regex, line, re.MULTILINE)
                for match in matches:
                    supertag = match.group(1)

                    if not supertag[0] in skipped_supertags:
                        supertags_here.append({"word": match.group(4), "supertags": [{"tag": match.group(1), "score": 1}]})

# write to JSON
with open("supertags.json", "w") as out:
    json.dump(supertags, out)


# write to EasyCCG "supertagged" format
# with open("supertagged.txt", "w") as out:
#     for sentence in supertags:
#         first = True
#         for word in sentence:
#             if first:
#                 first = False
#             else:
#                 out.write("|")

#             out.write(word["word"])
#             for supertag in word["supertags"]:
#                 out.write("\t")
#                 out.write(supertag["tag"])
#                 out.write("\t")
#                 out.write(str(supertag["score"]))

#         out.write("\n")


print(f"{len(supertags)} sentences converted.")


