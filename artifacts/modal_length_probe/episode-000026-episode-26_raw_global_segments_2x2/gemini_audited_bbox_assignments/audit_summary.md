# Gemini Audited BBox Assignments

Sheets: 23/23 ok
Boxes: 104
NoCharacter: 7
Changed from stable cleanup: 20

## Decision Counts
- keep_prior: 80
- correct_label: 19
- drop_not_character: 5

## Label Counts
- red-haired woman in red dress: 32
- white-haired injured man: 15
- woman in orange dress: 10
- blond boy in light robe: 8
- NoCharacter: 7
- red-haired man in brown suit: 7
- older woman in tan dress: 6
- brown-haired man with glasses in white shirt: 5
- middle-aged hotel manager in black suit: 4
- older white-haired bearded man: 3
- pink-haired person on screen: 2
- blond woman in brown jacket: 2
- dark-haired man in car: 1
- pink-haired person in light coat: 1
- woman in pink dress: 1

## Key Findings
- sheet_004_segments_009-011 old failure cause: this was not a Magi detection failure. Magi found bbox3 around the close-up person. The earlier free-form Gemini label was mixed (`red-haired woman in red dress and woman with brown hair in orange dress`), and the later stable cleanup mapped that mixed label to `NoCharacter`. That cleanup rule was too blunt.
- sheet_004_segments_009-011 bbox1: stable `NoCharacter` -> audited `woman in orange dress` (correct_label). Visible person in an orange dress and white mask.
- sheet_004_segments_009-011 bbox2: stable `woman in orange dress` -> audited `red-haired woman in red dress` (correct_label). Visible person with red hair in a red dress and white mask.
- sheet_004_segments_009-011 bbox3: stable `NoCharacter` -> audited `woman in orange dress` (correct_label). Close up of the woman in the orange dress and white mask.
- sheet_006_segments_015-016 old failure cause: Magi found plausible boxes, but the old label pass confused partial/overlapping boxes. The stricter audited pass treats Magi boxes as defaults, then uses Gemini to correct labels and drops animal/object-only boxes.
- sheet_006_segments_015-016 bbox5: stable `blond boy in light robe` -> audited `NoCharacter` (drop_not_character). The box contains only an animal (a bird) and gloved hands, not an isolated human character.
- sheet_006_segments_015-016 bbox6: stable `NoCharacter` -> audited `blond boy in light robe` (correct_label). This box clearly contains the blond boy reaching for the bird, previously unlabeled.
- sheet_006_segments_015-016 bbox8: stable `blond boy in light robe` -> audited `older woman in tan dress` (correct_label). The box contains the older woman in a tan dress, not a blond boy as suggested by previous labels.
- sheet_006_segments_015-016 bbox9: stable `red-haired woman in red dress` -> audited `blond boy in light robe` (correct_label). The box contains the blond boy wrapped in a blanket/robe, not the red-haired woman as suggested by previous labels.

## NoCharacter Audit
- sheet_004_segments_009-011 bbox4 (keep_prior): A small body fragment (top of a head) with no usable identity.
- sheet_006_segments_015-016 bbox5 (drop_not_character): The box contains only an animal (a bird) and gloved hands, not an isolated human character.
- sheet_007_segments_017-018 bbox3 (drop_not_character): This box is a crop of the object (bundle/bread) being held by the boy; it is not a separate character.
- sheet_008_segments_019-020 bbox3 (drop_not_character): This box focuses on a sub-region (hands and an animal) and overlaps with the primary character box bbox1.
- sheet_008_segments_019-020 bbox6 (drop_not_character): This box focuses on a sub-region (hands and an animal) and overlaps with the primary character box bbox4.
- sheet_014_segments_034-035 bbox1 (drop_not_character): This is a body fragment showing only a hand and sleeve, which does not provide a usable visual identity on its own.
- sheet_017_segments_040-042 bbox3 (keep_prior): This box contains only a hand holding keys, which is a body fragment with no usable identity.
