import csv

mapping = {}

with open("raw_districts.tsv", "r", encoding="utf-8", newline="") as infile:
    reader = csv.DictReader(infile, delimiter="\t")

    for row in reader:
        nation = row["nation"].strip()
        county = row["county"].strip()
        council = row["council"].strip()
        districts = row["districts"].strip()

        if council == "(district council)" or districts.lower() == "none":
            mapping[county] = nation
        else:
            for district in districts.split(","):
                mapping[district.strip()] = nation

def lookup_district(name):

    if name in mapping:
        return name, mapping[name]

    special_cases = {
        "Darlington": "England",
        "Folkestone and Hythe": "England",
        "Rhondda Cynon Taf": "Wales",
        "Salford": "England",
        "Westminster+City of London": "England",
        "Winchester": "England",
        "Na h-Eileanan Siar": "Scotland"
    }

    if name in special_cases:
        return name, special_cases[name]

    candidates = []

    if name.lower().startswith("city of "):
        candidates.append(name[8:].strip())

    if name.lower().endswith(" city"):
        candidates.append(name[:-5].strip())

    if name.startswith("The "):
        candidates.append(name[4:].strip())

    if "," in name:
        candidates.append(name.split(",", 1)[0].strip())

    if name.lower().endswith(" upon thames"):
        return name, "England"

    for candidate in candidates:
        if candidate in mapping:
            return candidate, mapping[candidate]

    return None, None

with open("commute.csv", "r", encoding="utf-8", newline="") as infile:
    reader = csv.reader(infile)
    header = next(reader)

commute_districts = [d.strip() for d in header[1:]]

missing = []

with open("crosswalk.tsv", "w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile, delimiter="\t")
    writer.writerow(["district", "nation"])

    for district in commute_districts:
        matched_name, nation = lookup_district(district)

        if nation is None:
            missing.append(district)
        else:
            writer.writerow([district, nation])

print(f"{len(commute_districts)} districts found in commute.csv")
print(f"{len(commute_districts) - len(missing)} matched")
print(f"{len(missing)} missing")

if missing:
    print("\nUnmapped districts:")
    for district in sorted(missing):
        print(district)
