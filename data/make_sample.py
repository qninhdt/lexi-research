"""Create the 100 hand-authored training samples used for local SFT checks."""

# The static JSON intentionally keeps each sample legible as one record.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FIELDS = (
    "target",
    "definition",
    "pos",
    "text",
    "correction",
    "meaning",
    "feedback",
)

# Four independently authored batches, five examples for each of twenty words.
ROWS = json.loads(
    r'''[
  {
    "target": "advise",
    "definition": "to give someone a recommendation about what they should do",
    "pos": "verb",
    "text": "The nurse advised me to drink more water while I recovered from the fever.",
    "correction": "The nurse advised me to drink more water while I recovered from the fever.",
    "meaning": 4,
    "feedback": "Advised gives a recommendation about what the person should do."
  },
  {
    "target": "advise",
    "definition": "to give someone a recommendation about what they should do",
    "pos": "verb",
    "text": "My mentor advised that I apply for the scholarship before Friday.",
    "correction": "My mentor advised that I apply for the scholarship before Friday.",
    "meaning": 4,
    "feedback": "Advised recommends an action."
  },
  {
    "target": "advise",
    "definition": "to give someone a recommendation about what they should do",
    "pos": "verb",
    "text": "The guide advise us to carry a map.",
    "correction": "The guide [advise>advised:tense] us to carry a map.",
    "meaning": 4,
    "feedback": "The recommendation is correct, but the past-tense verb should be advised."
  },
  {
    "target": "advise",
    "definition": "to give someone a recommendation about what they should do",
    "pos": "verb",
    "text": "The doctor advised me rest for two days.",
    "correction": "The doctor advised me [rest>to rest:part] for two days.",
    "meaning": 4,
    "feedback": "After advised me, use to rest to introduce the recommended action."
  },
  {
    "target": "advise",
    "definition": "to give someone a recommendation about what they should do",
    "pos": "verb",
    "text": "The email advised us that the meeting had moved to Monday.",
    "correction": null,
    "meaning": 0,
    "feedback": "Here advised means informed, not recommended an action."
  },
  {
    "target": "borrow",
    "definition": "to take and use something belonging to another person with the intention of returning it",
    "pos": "verb",
    "text": "Can I borrow your charger until lunchtime?",
    "correction": "Can I borrow your charger until lunchtime?",
    "meaning": 4,
    "feedback": "The speaker wants to use someone else's charger and return it later."
  },
  {
    "target": "borrow",
    "definition": "to take and use something belonging to another person with the intention of returning it",
    "pos": "verb",
    "text": "Mai borrowed a ladder from her neighbor to hang a picture.",
    "correction": "Mai borrowed a ladder from her neighbor to hang a picture.",
    "meaning": 4,
    "feedback": "Borrowed describes temporary use of the neighbor's ladder."
  },
  {
    "target": "borrow",
    "definition": "to take and use something belonging to another person with the intention of returning it",
    "pos": "verb",
    "text": "I borrow a novel from the library last week.",
    "correction": "I [borrow>borrowed:tense] a novel from the library last week.",
    "meaning": 4,
    "feedback": "The sentence describes taking a library book temporarily; use the past tense borrowed."
  },
  {
    "target": "borrow",
    "definition": "to take and use something belonging to another person with the intention of returning it",
    "pos": "verb",
    "text": "They borrowed a tent to their friends for the hike.",
    "correction": "They borrowed a tent [to>from:prep] their friends for the hike.",
    "meaning": 4,
    "feedback": "Use from to identify the owner of the borrowed tent."
  },
  {
    "target": "borrow",
    "definition": "to take and use something belonging to another person with the intention of returning it",
    "pos": "verb",
    "text": "The company borrowed heavily from classical mythology for its logo.",
    "correction": null,
    "meaning": 0,
    "feedback": "Borrowed here means adapted ideas from mythology, with no intention of returning an object."
  },
  {
    "target": "discover",
    "definition": "to find or learn something for the first time",
    "pos": "verb",
    "text": "The children discovered a hidden path behind the waterfall.",
    "correction": "The children discovered a hidden path behind the waterfall.",
    "meaning": 4,
    "feedback": "Discovered means found the path for the first time."
  },
  {
    "target": "discover",
    "definition": "to find or learn something for the first time",
    "pos": "verb",
    "text": "After checking the archives, we discovered that the building was over a century old.",
    "correction": "After checking the archives, we discovered that the building was over a century old.",
    "meaning": 4,
    "feedback": "Discovered means learned this fact for the first time."
  },
  {
    "target": "discover",
    "definition": "to find or learn something for the first time",
    "pos": "verb",
    "text": "She discover an old photograph in the attic.",
    "correction": "She [discover>discovered:tense] an old photograph in the attic.",
    "meaning": 4,
    "feedback": "The sentence describes finding something new; use the past tense discovered."
  },
  {
    "target": "discover",
    "definition": "to find or learn something for the first time",
    "pos": "verb",
    "text": "The scientist hopes discover a safer treatment.",
    "correction": "The scientist hopes [discover>to discover:part] a safer treatment.",
    "meaning": 4,
    "feedback": "After hopes, use the infinitive to discover."
  },
  {
    "target": "discover",
    "definition": "to find or learn something for the first time",
    "pos": "verb",
    "text": "I discovered the same shortcut again on my way home.",
    "correction": null,
    "meaning": 0,
    "feedback": "Again shows that the shortcut was already known, so this is not a first discovery."
  },
  {
    "target": "explain",
    "definition": "to make an idea or situation clear by describing it",
    "pos": "verb",
    "text": "Could you explain the new safety procedure to the interns?",
    "correction": "Could you explain the new safety procedure to the interns?",
    "meaning": 4,
    "feedback": "Explain is used to make the safety procedure clear."
  },
  {
    "target": "explain",
    "definition": "to make an idea or situation clear by describing it",
    "pos": "verb",
    "text": "Her diagram explained how the water filter works.",
    "correction": "Her diagram explained how the water filter works.",
    "meaning": 4,
    "feedback": "The diagram makes the filter's operation clear."
  },
  {
    "target": "explain",
    "definition": "to make an idea or situation clear by describing it",
    "pos": "verb",
    "text": "The tutor explain the equation in three different ways.",
    "correction": "The tutor [explain>explains:agr] the equation in three different ways.",
    "meaning": 4,
    "feedback": "The singular subject tutor requires explains."
  },
  {
    "target": "explain",
    "definition": "to make an idea or situation clear by describing it",
    "pos": "verb",
    "text": "Please explain me why the schedule changed.",
    "correction": "Please explain [me>to me:part] why the schedule changed.",
    "meaning": 4,
    "feedback": "Use explain something to someone, so to me is required."
  },
  {
    "target": "explain",
    "definition": "to make an idea or situation clear by describing it",
    "pos": "verb",
    "text": "The witness explained away the broken window by blaming the wind.",
    "correction": null,
    "meaning": 0,
    "feedback": "Explained away means dismissed or justified the problem, not simply described it to make it clear."
  },
  {
    "target": "repair",
    "definition": "to fix something that is damaged or not working",
    "pos": "verb",
    "text": "Can you repair this broken zipper before the trip?",
    "correction": "Can you repair this broken zipper before the trip?",
    "meaning": 4,
    "feedback": "Repair means fix the damaged zipper."
  },
  {
    "target": "repair",
    "definition": "to fix something that is damaged or not working",
    "pos": "verb",
    "text": "The technician repaired the leaking pipe before the kitchen flooded.",
    "correction": "The technician repaired the leaking pipe before the kitchen flooded.",
    "meaning": 4,
    "feedback": "Repaired means fixed the damaged pipe."
  },
  {
    "target": "repair",
    "definition": "to fix something that is damaged or not working",
    "pos": "verb",
    "text": "The mechanic repair my scooter yesterday.",
    "correction": "The mechanic [repair>repaired:tense] my scooter yesterday.",
    "meaning": 4,
    "feedback": "The sentence describes a completed fix, so use repaired."
  },
  {
    "target": "repair",
    "definition": "to fix something that is damaged or not working",
    "pos": "verb",
    "text": "The technician agreed repair the faulty printer.",
    "correction": "The technician agreed [repair>to repair:part] the faulty printer.",
    "meaning": 4,
    "feedback": "After agreed, use the infinitive to repair."
  },
  {
    "target": "repair",
    "definition": "to fix something that is damaged or not working",
    "pos": "verb",
    "text": "The apology repaired their friendship after the argument.",
    "correction": null,
    "meaning": 0,
    "feedback": "Repair refers to restoring a relationship here, not fixing a damaged or malfunctioning thing."
  },
  {
    "target": "ancient",
    "definition": "belonging to a very distant period of history",
    "pos": "adjective",
    "text": "The museum displays an ancient Egyptian vase from a royal tomb.",
    "correction": "The museum displays an ancient Egyptian vase from a royal tomb.",
    "meaning": 4,
    "feedback": "Correct: the vase belongs to a distant historical period."
  },
  {
    "target": "ancient",
    "definition": "belonging to a very distant period of history",
    "pos": "adjective",
    "text": "These ancient ruins was buried under sand for centuries.",
    "correction": "These ancient ruins [was>were:agr] buried under sand for centuries.",
    "meaning": 4,
    "feedback": "The meaning is correct, but the plural subject requires were."
  },
  {
    "target": "ancient",
    "definition": "belonging to a very distant period of history",
    "pos": "adjective",
    "text": "Researchers discovered an ancient manuscript in the monastery's library.",
    "correction": "Researchers discovered an ancient manuscript in the monastery's library.",
    "meaning": 4,
    "feedback": "Correct: the manuscript belongs to a distant historical period."
  },
  {
    "target": "ancient",
    "definition": "belonging to a very distant period of history",
    "pos": "adjective",
    "text": "My ancient laptop still runs the latest games.",
    "correction": null,
    "meaning": 0,
    "feedback": "This describes an old modern device, not something belonging to a distant historical period."
  },
  {
    "target": "ancient",
    "definition": "belonging to a very distant period of history",
    "pos": "adjective",
    "text": "The ancient carvings has survived thousands of years.",
    "correction": "The ancient carvings [has>have:agr] survived thousands of years.",
    "meaning": 4,
    "feedback": "The meaning is correct, but the plural subject requires have."
  },
  {
    "target": "narrow",
    "definition": "having a small distance from one side to the other",
    "pos": "adjective",
    "text": "We walked along a narrow path between the cliffs.",
    "correction": "We walked along a narrow path between the cliffs.",
    "meaning": 4,
    "feedback": "Correct: the path has little width from side to side."
  },
  {
    "target": "narrow",
    "definition": "having a small distance from one side to the other",
    "pos": "adjective",
    "text": "This narrow bridge can hold only one car at a time.",
    "correction": "This narrow bridge can hold only one car at a time.",
    "meaning": 4,
    "feedback": "Correct: the bridge has a small width."
  },
  {
    "target": "narrow",
    "definition": "having a small distance from one side to the other",
    "pos": "adjective",
    "text": "The hallway is too narrow for move the sofa through.",
    "correction": "The hallway is too narrow [for>to:prep] move the sofa through.",
    "meaning": 4,
    "feedback": "The meaning is correct; too narrow is followed by an infinitive with to."
  },
  {
    "target": "narrow",
    "definition": "having a small distance from one side to the other",
    "pos": "adjective",
    "text": "The narrow alley lead to the market.",
    "correction": "The narrow alley [lead>leads:agr] to the market.",
    "meaning": 4,
    "feedback": "The meaning is correct, but the singular subject requires leads."
  },
  {
    "target": "narrow",
    "definition": "having a small distance from one side to the other",
    "pos": "adjective",
    "text": "He took a narrow view of the complicated problem.",
    "correction": null,
    "meaning": 0,
    "feedback": "This describes a limited viewpoint, not physical width."
  },
  {
    "target": "patient",
    "definition": "able to wait calmly without becoming annoyed",
    "pos": "adjective",
    "text": "The patient teacher waited calmly while every student finished the quiz.",
    "correction": "The patient teacher waited calmly while every student finished the quiz.",
    "meaning": 4,
    "feedback": "Correct: the teacher remains calm while waiting."
  },
  {
    "target": "patient",
    "definition": "able to wait calmly without becoming annoyed",
    "pos": "adjective",
    "text": "Please be patient while the technician repairs the printer.",
    "correction": "Please be patient while the technician repairs the printer.",
    "meaning": 4,
    "feedback": "Correct: the speaker asks for calm waiting."
  },
  {
    "target": "patient",
    "definition": "able to wait calmly without becoming annoyed",
    "pos": "adjective",
    "text": "Mina was patient enough to waited for her little brother.",
    "correction": "Mina was patient enough to [waited>wait:form] for her little brother.",
    "meaning": 4,
    "feedback": "The meaning is correct, but an infinitive follows enough to."
  },
  {
    "target": "patient",
    "definition": "able to wait calmly without becoming annoyed",
    "pos": "adjective",
    "text": "The patient passengers was waiting quietly despite the long delay.",
    "correction": "The patient passengers [was>were:agr] waiting quietly despite the long delay.",
    "meaning": 4,
    "feedback": "The meaning is correct, but the plural subject requires were."
  },
  {
    "target": "patient",
    "definition": "able to wait calmly without becoming annoyed",
    "pos": "adjective",
    "text": "The doctor examined the patient before prescribing medicine.",
    "correction": null,
    "meaning": 0,
    "feedback": "Here the word names someone receiving medical care rather than describing calm waiting."
  },
  {
    "target": "shelter",
    "definition": "a place that gives protection from weather or danger",
    "pos": "noun",
    "text": "The hikers found shelter under a rocky overhang during the storm.",
    "correction": "The hikers found shelter under a rocky overhang during the storm.",
    "meaning": 4,
    "feedback": "Correct: the overhang protects the hikers from the storm."
  },
  {
    "target": "shelter",
    "definition": "a place that gives protection from weather or danger",
    "pos": "noun",
    "text": "The animal shelter gives abandoned dogs a safe place to sleep.",
    "correction": "The animal shelter gives abandoned dogs a safe place to sleep.",
    "meaning": 4,
    "feedback": "Correct: the facility provides protection for abandoned animals."
  },
  {
    "target": "shelter",
    "definition": "a place that gives protection from weather or danger",
    "pos": "noun",
    "text": "The town opened shelter for families after the flood.",
    "correction": "The town opened [shelter>a shelter:art] for families after the flood.",
    "meaning": 4,
    "feedback": "The meaning is correct, but a singular count noun needs the article a."
  },
  {
    "target": "shelter",
    "definition": "a place that gives protection from weather or danger",
    "pos": "noun",
    "text": "Emergency workers provided shelter for families displaced by the flood.",
    "correction": "Emergency workers provided shelter for families displaced by the flood.",
    "meaning": 4,
    "feedback": "Correct: the workers offered a protective place after the flood."
  },
  {
    "target": "shelter",
    "definition": "a place that gives protection from weather or danger",
    "pos": "noun",
    "text": "Her umbrella can shelter us from the rain.",
    "correction": null,
    "meaning": 0,
    "feedback": "The word is used as a verb meaning protect, not as a protective place."
  },
  {
    "target": "journey",
    "definition": "an act of travelling from one place to another",
    "pos": "noun",
    "text": "Our journey across the desert lasted six days.",
    "correction": "Our journey across the desert lasted six days.",
    "meaning": 4,
    "feedback": "Correct: this names physical travel across the desert."
  },
  {
    "target": "journey",
    "definition": "an act of travelling from one place to another",
    "pos": "noun",
    "text": "The train journey from Hanoi to Hue was peaceful.",
    "correction": "The train journey from Hanoi to Hue was peaceful.",
    "meaning": 4,
    "feedback": "Correct: this names travel by train between two places."
  },
  {
    "target": "journey",
    "definition": "an act of travelling from one place to another",
    "pos": "noun",
    "text": "The journey were longer than we expected.",
    "correction": "The journey [were>was:agr] longer than we expected.",
    "meaning": 4,
    "feedback": "The meaning is correct, but singular journey requires was."
  },
  {
    "target": "journey",
    "definition": "an act of travelling from one place to another",
    "pos": "noun",
    "text": "Their journey through the mountains become dangerous after sunset.",
    "correction": "Their journey through the mountains [become>became:tense] dangerous after sunset.",
    "meaning": 4,
    "feedback": "The meaning is correct, but the past-tense verb should be became."
  },
  {
    "target": "journey",
    "definition": "an act of travelling from one place to another",
    "pos": "noun",
    "text": "She described her recovery as a long journey.",
    "correction": null,
    "meaning": 0,
    "feedback": "Recovery is described metaphorically here, not as travel between places."
  },
  {
    "target": "analyze",
    "definition": "to examine something carefully in order to understand it",
    "pos": "verb",
    "text": "The researchers analyzed the survey responses to understand why customers left.",
    "correction": "The researchers analyzed the survey responses to understand why customers left.",
    "meaning": 4,
    "feedback": "Correct use: the researchers examined the responses carefully to understand the customers' decisions."
  },
  {
    "target": "analyze",
    "definition": "to examine something carefully in order to understand it",
    "pos": "verb",
    "text": "Before the meeting, Maya analyze the budget reports.",
    "correction": "Before the meeting, Maya [analyze>analyzed:tense] the budget reports.",
    "meaning": 4,
    "feedback": "The intended meaning is correct, but the past-tense verb is needed."
  },
  {
    "target": "analyze",
    "definition": "to examine something carefully in order to understand it",
    "pos": "verb",
    "text": "Our team will analyze the soil samples before choosing a crop.",
    "correction": "Our team will analyze the soil samples before choosing a crop.",
    "meaning": 4,
    "feedback": "Correct use: the team will examine the samples carefully before making a decision."
  },
  {
    "target": "analyze",
    "definition": "to examine something carefully in order to understand it",
    "pos": "verb",
    "text": "The teacher analyzed the students into four groups for the activity.",
    "correction": null,
    "meaning": 0,
    "feedback": "Wrong sense: the verb is being used as if it meant sort or divide."
  },
  {
    "target": "analyze",
    "definition": "to examine something carefully in order to understand it",
    "pos": "verb",
    "text": "The software analyze every paragraph for repeated ideas.",
    "correction": "The software [analyze>analyzes:agr] every paragraph for repeated ideas.",
    "meaning": 4,
    "feedback": "The intended meaning is correct, but a singular subject requires the third-person singular form."
  },
  {
    "target": "deadline",
    "definition": "the latest time or date by which something must be completed",
    "pos": "noun",
    "text": "The final deadline for submitting the application is October 18.",
    "correction": "The final deadline for submitting the application is October 18.",
    "meaning": 4,
    "feedback": "Correct use: this refers to the latest date for completing the application."
  },
  {
    "target": "deadline",
    "definition": "the latest time or date by which something must be completed",
    "pos": "noun",
    "text": "We missed the deadline for submit the report.",
    "correction": "We missed the deadline for [submit>submitting:part] the report.",
    "meaning": 4,
    "feedback": "The intended meaning is correct, but a gerund is required after the preposition."
  },
  {
    "target": "deadline",
    "definition": "the latest time or date by which something must be completed",
    "pos": "noun",
    "text": "Set a reminder so that you finish the form before the deadline.",
    "correction": "Set a reminder so that you finish the form before the deadline.",
    "meaning": 4,
    "feedback": "Correct use: the word identifies the final time allowed for finishing the form."
  },
  {
    "target": "deadline",
    "definition": "the latest time or date by which something must be completed",
    "pos": "noun",
    "text": "The deadline between the two farms runs along the old stone wall.",
    "correction": null,
    "meaning": 0,
    "feedback": "Wrong sense: a line separating properties is a boundary, not a final completion time."
  },
  {
    "target": "deadline",
    "definition": "the latest time or date by which something must be completed",
    "pos": "noun",
    "text": "The deadline for these applications are Monday.",
    "correction": "The deadline for these applications [are>is:agr] Monday.",
    "meaning": 4,
    "feedback": "The intended meaning is correct, but the singular subject requires is."
  },
  {
    "target": "efficient",
    "definition": "working well without wasting time, effort, or resources",
    "pos": "adjective",
    "text": "The efficient irrigation system delivers enough water while using very little electricity.",
    "correction": "The efficient irrigation system delivers enough water while using very little electricity.",
    "meaning": 4,
    "feedback": "Correct use: the system works well while conserving electricity and water."
  },
  {
    "target": "efficient",
    "definition": "working well without wasting time, effort, or resources",
    "pos": "adjective",
    "text": "The new assistant organizes the files efficient.",
    "correction": "The new assistant organizes the files [efficient>efficiently:form].",
    "meaning": 4,
    "feedback": "The intended meaning is correct, but an adverb is needed to describe how the assistant organizes the files."
  },
  {
    "target": "efficient",
    "definition": "working well without wasting time, effort, or resources",
    "pos": "adjective",
    "text": "The medicine was efficient at relieving pain, even though it required a large dose and several hours to work.",
    "correction": null,
    "meaning": 0,
    "feedback": "Wrong sense: this uses the word to mean simply effective rather than resource-saving."
  },
  {
    "target": "efficient",
    "definition": "working well without wasting time, effort, or resources",
    "pos": "adjective",
    "text": "Using a shared spreadsheet is an efficient way to track the team's expenses.",
    "correction": "Using a shared spreadsheet is an efficient way to track the team's expenses.",
    "meaning": 4,
    "feedback": "Correct use: the spreadsheet tracks expenses with little wasted time or effort."
  },
  {
    "target": "efficient",
    "definition": "working well without wasting time, effort, or resources",
    "pos": "adjective",
    "text": "This route is efficienter than the old one.",
    "correction": "This route is [efficienter>more efficient:form] than the old one.",
    "meaning": 4,
    "feedback": "The intended meaning is correct, but the comparative form is more efficient."
  },
  {
    "target": "negotiate",
    "definition": "to discuss something formally in order to reach an agreement",
    "pos": "verb",
    "text": "The union and the company negotiated a new contract after three long meetings.",
    "correction": "The union and the company negotiated a new contract after three long meetings.",
    "meaning": 4,
    "feedback": "Correct use: both sides formally discussed the contract to reach an agreement."
  },
  {
    "target": "negotiate",
    "definition": "to discuss something formally in order to reach an agreement",
    "pos": "verb",
    "text": "We need negotiate the price before signing the contract.",
    "correction": "We [need negotiate>need to negotiate:part] the price before signing the contract.",
    "meaning": 4,
    "feedback": "The intended meaning is correct, but need is followed by to and the base verb."
  },
  {
    "target": "negotiate",
    "definition": "to discuss something formally in order to reach an agreement",
    "pos": "verb",
    "text": "The hikers negotiated a steep pass before sunset.",
    "correction": null,
    "meaning": 0,
    "feedback": "Wrong sense: here the verb means travel through a difficult route, not formally discuss an agreement."
  },
  {
    "target": "negotiate",
    "definition": "to discuss something formally in order to reach an agreement",
    "pos": "verb",
    "text": "The diplomats negotiated a cease-fire after several private meetings.",
    "correction": "The diplomats negotiated a cease-fire after several private meetings.",
    "meaning": 4,
    "feedback": "Correct use: the diplomats formally discussed terms to reach an agreement."
  },
  {
    "target": "negotiate",
    "definition": "to discuss something formally in order to reach an agreement",
    "pos": "verb",
    "text": "The buyer negotiated the seller about the delivery date.",
    "correction": "The buyer [negotiated the seller>negotiated with the seller:prep] about the delivery date.",
    "meaning": 4,
    "feedback": "The intended meaning is correct, but the person involved in the discussion is introduced with with."
  },
  {
    "target": "reliable",
    "definition": "able to be trusted to work well or behave as expected",
    "pos": "adjective",
    "text": "Our reliable internet connection rarely drops during video calls.",
    "correction": "Our reliable internet connection rarely drops during video calls.",
    "meaning": 4,
    "feedback": "Correct use: the connection can be trusted to work consistently."
  },
  {
    "target": "reliable",
    "definition": "able to be trusted to work well or behave as expected",
    "pos": "adjective",
    "text": "The emergency generator are reliable even in freezing weather.",
    "correction": "The emergency generator [are>is:agr] reliable even in freezing weather.",
    "meaning": 4,
    "feedback": "The intended meaning is correct, but the singular subject requires is."
  },
  {
    "target": "reliable",
    "definition": "able to be trusted to work well or behave as expected",
    "pos": "adjective",
    "text": "She chose a reliable color for the logo because blue felt safe.",
    "correction": null,
    "meaning": 0,
    "feedback": "Wrong sense: the adjective describes something dependable, not a color chosen because it feels safe."
  },
  {
    "target": "reliable",
    "definition": "able to be trusted to work well or behave as expected",
    "pos": "adjective",
    "text": "The backup server is reliable enough to run continuously for months.",
    "correction": "The backup server is reliable enough to run continuously for months.",
    "meaning": 4,
    "feedback": "Correct use: the server can be trusted to work continuously as expected."
  },
  {
    "target": "reliable",
    "definition": "able to be trusted to work well or behave as expected",
    "pos": "adjective",
    "text": "The courier delivered every package reliable during the holiday rush.",
    "correction": "The courier delivered every package [reliable>reliably:form] during the holiday rush.",
    "meaning": 4,
    "feedback": "The intended meaning is correct, but an adverb is needed to describe how the packages were delivered."
  },
  {
    "target": "apologize",
    "definition": "to say that you are sorry for doing something wrong",
    "pos": "verb",
    "text": "I apologize for arrive late to the interview.",
    "correction": "I apologize for [arrive>arriving:form] late to the interview.",
    "meaning": 4,
    "feedback": "The sentence uses apologize for the intended apology sense, but for must be followed by an -ing form."
  },
  {
    "target": "apologize",
    "definition": "to say that you are sorry for doing something wrong",
    "pos": "verb",
    "text": "The airline's notice apologize for the cancellation, although no employee signed it.",
    "correction": "The airline's notice [apologize>apologizes:agr] for the cancellation, although no employee signed it.",
    "meaning": 3,
    "feedback": "The sentence uses the intended apology sense indirectly, but the singular subject needs apologizes."
  },
  {
    "target": "apologize",
    "definition": "to say that you are sorry for doing something wrong",
    "pos": "verb",
    "text": "I apologize, but the last train has already left.",
    "correction": "I apologize, but the last train has already left.",
    "meaning": 2,
    "feedback": "This expresses regret about bad news rather than clearly apologizing for a wrong action."
  },
  {
    "target": "apologize",
    "definition": "to say that you are sorry for doing something wrong",
    "pos": "verb",
    "text": "The editorial apologized for the mayor's decision, insisting that the emergency left no alternative.",
    "correction": "The editorial apologized for the mayor's decision, insisting that the emergency left no alternative.",
    "meaning": 1,
    "feedback": "The editorial uses apologize to defend or justify a decision, a near sense rather than a clear apology."
  },
  {
    "target": "apologize",
    "definition": "to say that you are sorry for doing something wrong",
    "pos": "verb",
    "text": "The teacher asked us to write an apologize at the end of the email.",
    "correction": null,
    "meaning": 0,
    "feedback": "Here apologize is used as a noun for a written message, not as the verb meaning to say sorry."
  },
  {
    "target": "celebrate",
    "definition": "to do something enjoyable to mark a special occasion or achievement",
    "pos": "verb",
    "text": "Last night, we celebrate our coach's retirement with homemade pizza.",
    "correction": "Last night, we [celebrate>celebrated:tense] our coach's retirement with homemade pizza.",
    "meaning": 4,
    "feedback": "The meaning is correct, but the past-tense form celebrated is needed."
  },
  {
    "target": "celebrate",
    "definition": "to do something enjoyable to mark a special occasion or achievement",
    "pos": "verb",
    "text": "The lab celebrated the successful launch by sharing a photo on its newsletter.",
    "correction": "The lab celebrated the successful launch by sharing a photo [on>in:prep] its newsletter.",
    "meaning": 3,
    "feedback": "The lab marks an achievement indirectly, but photos are shared in a newsletter."
  },
  {
    "target": "celebrate",
    "definition": "to do something enjoyable to mark a special occasion or achievement",
    "pos": "verb",
    "text": "The gallery celebrated its retiring director with a portrait in the lobby.",
    "correction": "The gallery celebrated its retiring director with a portrait in the lobby.",
    "meaning": 2,
    "feedback": "This could mean marking the retirement with a tribute or simply praising the director, so the sense is ambiguous."
  },
  {
    "target": "celebrate",
    "definition": "to do something enjoyable to mark a special occasion or achievement",
    "pos": "verb",
    "text": "The newspaper celebrated the mayor's honesty in a glowing editorial.",
    "correction": "The newspaper celebrated the mayor's honesty in a glowing editorial.",
    "meaning": 1,
    "feedback": "Here celebrate means praise rather than do something enjoyable to mark an occasion."
  },
  {
    "target": "celebrate",
    "definition": "to do something enjoyable to mark a special occasion or achievement",
    "pos": "verb",
    "text": "The nurse celebrated the patient's temperature on a chart before calling the doctor.",
    "correction": null,
    "meaning": 0,
    "feedback": "Here celebrate is used as if it meant record, which is unrelated to marking an occasion."
  },
  {
    "target": "crowded",
    "definition": "full of people, leaving little available space",
    "pos": "adjective",
    "text": "The station were crowded after the concert ended.",
    "correction": "The station [were>was:agr] crowded after the concert ended.",
    "meaning": 4,
    "feedback": "The sentence describes a place full of people, but the singular subject needs was."
  },
  {
    "target": "crowded",
    "definition": "full of people, leaving little available space",
    "pos": "adjective",
    "text": "The small cafe was crowded with regulars, yet we found two stools by the window.",
    "correction": "The small cafe was crowded with regulars, yet we found two stools by the window.",
    "meaning": 3,
    "feedback": "The cafe is mostly full of people, though two empty stools add a small nuance."
  },
  {
    "target": "crowded",
    "definition": "full of people, leaving little available space",
    "pos": "adjective",
    "text": "The bus stop was crowded with luggage and a few travelers, leaving barely any room to stand.",
    "correction": "The bus stop was crowded with luggage and a few travelers, leaving barely any room to stand.",
    "meaning": 2,
    "feedback": "The lack of space comes mainly from luggage rather than many people, so the sense is ambiguous."
  },
  {
    "target": "crowded",
    "definition": "full of people, leaving little available space",
    "pos": "adjective",
    "text": "The instructions were crowded with tiny details, so the new staff found them hard to scan.",
    "correction": "The instructions were crowded with tiny details, so the new staff found them hard to scan.",
    "meaning": 1,
    "feedback": "Here crowded means packed with information rather than full of people."
  },
  {
    "target": "crowded",
    "definition": "full of people, leaving little available space",
    "pos": "adjective",
    "text": "The fans crowded the interns into a narrow doorway during the evacuation.",
    "correction": null,
    "meaning": 0,
    "feedback": "Here crowded is a past-tense verb meaning pressed people together, not the adjective."
  },
  {
    "target": "neighborhood",
    "definition": "an area of a town where people live",
    "pos": "noun",
    "text": "The neighborhood have a small library and a weekend farmers' market.",
    "correction": "The neighborhood [have>has:agr] a small library and a weekend farmers' market.",
    "meaning": 4,
    "feedback": "The sentence uses neighborhood for a residential area, but the singular subject needs has."
  },
  {
    "target": "neighborhood",
    "definition": "an area of a town where people live",
    "pos": "noun",
    "text": "The neighborhood welcomed new family with a shared meal.",
    "correction": "The neighborhood welcomed [>a:art] new family with a shared meal.",
    "meaning": 3,
    "feedback": "The area sense is clear through the residents' actions, but the noun needs the article a."
  },
  {
    "target": "neighborhood",
    "definition": "an area of a town where people live",
    "pos": "noun",
    "text": "The neighborhood decided to close the playground at dusk.",
    "correction": "The neighborhood decided to close the playground at dusk.",
    "meaning": 2,
    "feedback": "Neighborhood may mean the residents rather than the residential area itself, so the sense is ambiguous."
  },
  {
    "target": "neighborhood",
    "definition": "an area of a town where people live",
    "pos": "noun",
    "text": "The search algorithm examines each pixel's neighborhood before choosing a value.",
    "correction": "The search algorithm examines each pixel's neighborhood before choosing a value.",
    "meaning": 1,
    "feedback": "Here neighborhood means the surrounding area of a pixel, a near sense rather than a residential area."
  },
  {
    "target": "neighborhood",
    "definition": "an area of a town where people live",
    "pos": "noun",
    "text": "A neighborhood is the person who lives next door.",
    "correction": null,
    "meaning": 0,
    "feedback": "A neighborhood is an area, not a person who lives next door; that person is a neighbor."
  },
  {
    "target": "volunteer",
    "definition": "to offer to do something without being paid",
    "pos": "verb",
    "text": "I volunteer to sort the donated books at the shelter last Saturday.",
    "correction": "I [volunteer>volunteered:tense] to sort the donated books at the shelter last Saturday.",
    "meaning": 4,
    "feedback": "The sentence expresses an unpaid offer to help, but the past-tense form volunteered is needed."
  },
  {
    "target": "volunteer",
    "definition": "to offer to do something without being paid",
    "pos": "verb",
    "text": "He volunteered at the shelter every Tuesday as part of his community-service requirement.",
    "correction": "He volunteered at the shelter every Tuesday as part of his community-service requirement.",
    "meaning": 3,
    "feedback": "The work is unpaid, but the requirement makes the offer less fully voluntary."
  },
  {
    "target": "volunteer",
    "definition": "to offer to do something without being paid",
    "pos": "verb",
    "text": "When the manager asked for someone to present the quarterly figures, Jo volunteered.",
    "correction": "When the manager asked for someone to present the quarterly figures, Jo volunteered.",
    "meaning": 2,
    "feedback": "Jo offers to do the task, but the sentence does not show whether the work will be unpaid."
  },
  {
    "target": "volunteer",
    "definition": "to offer to do something without being paid",
    "pos": "verb",
    "text": "The nurse volunteered that the patient had missed a dose.",
    "correction": "The nurse volunteered that the patient had missed a dose.",
    "meaning": 1,
    "feedback": "Here volunteered means offered information rather than offered to do unpaid work."
  },
  {
    "target": "volunteer",
    "definition": "to offer to do something without being paid",
    "pos": "verb",
    "text": "The charity volunteer paid every helper for the hours they worked.",
    "correction": null,
    "meaning": 0,
    "feedback": "Here volunteer is a noun for an unpaid helper, not the requested verb."
  }
]'''
)

if len(ROWS) != 100:
    raise ValueError(f"expected 100 rows, got {len(ROWS)}")
if any(tuple(row) != FIELDS for row in ROWS):
    raise ValueError("sample rows do not have the expected fields")
if len({row["text"] for row in ROWS}) != len(ROWS):
    raise ValueError("sample rows contain duplicate text")

table = pa.table({field: [row[field] for row in ROWS] for field in FIELDS})
Path("data").mkdir(exist_ok=True)
pq.write_table(table, "data/sample_train.parquet")
print(f"Wrote {len(ROWS)} rows to data/sample_train.parquet")
