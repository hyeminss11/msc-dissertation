import json
import os

# 1. Put the rewritten content
entry = {
    "author": "mlevenson",
    "original": "...",
    "rewritten": "..."
}

# 2. json file directory
output_path = "experiments/llm_rewriting_trial1/output/articles_rewritten_manual.json"

# 3. load json
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = []

# 4. add new data and save
entry = {
    "author": "greenslader",
    "original": "Spotlight, the current affairs strand made by BBC Northern Ireland, has been garlanded with awards during its 43-year history. But I think it has lost the run of itself after seeing its programme a week ago about the 2006 killing of a Sinn Féin official, Denis Donaldson. His death followed soon after he had admitted to being a British agent for the best part of 20 years. Its central claim was that Gerry Adams, the party's leader, had been responsible for ordering the murder. And such is the status of Spotlight that the allegation made headlines in Ireland's leading newspapers, plus one or two UK nationals. The programme, Spy in the IRA, relied on the \"evidence\" of an anonymous single source. His identity and voice were concealed \"because of fears for his safety.\" He was said to be a former IRA volunteer and Sinn Féin member who had himself become an informer for the British secret services. His \"exclusive interview\" was hyped up by Spotlight's reporter, Jennifer O'Leary, in a pre-broadcast preview article in which she wrote: \"For months, I've been in contact with a man, referred to in the programme as ‘Martin'\". On screen, however, his statements lacked any semblance of truthfulness. They amounted to staccato answers to leading questions. There was not a scintilla of supporting evidence for his claims about Adams, who is a member of the Irish parliament, having sanctioned the murder. As a journalistic exercise, the programme looked to be wholly inadequate. And it didn't take long for its allegation about Adams's involvement to unravel. That wasn't too unsurprising since the Real IRA - one of the so-called dissident groups antagonistic to Sinn Féin - had admitted its responsibity for killing Donaldson some three years after his death. First, Adams issued a statement, reported on the BBC website, saying: \"I specifically and categorically refute these unsubstantiated allegations\", which he saw as an attempt to smear him. Second, Adams's solicitor followed up with another unequivocal rebuttal, as reported by the Guardian, about his client having \"no knowledge of and... no involvement whatsoever in the killing of Denis Donaldson.\" Third, the Guardian reported on Friday that Donaldson's relatives had rejected the Spotlight claim. A solicitor acting for them and speaking on their behalf called it \"absolute nonsense.\" Fourth, the clincher: the Sunday Times's Irish edition ran a story about the Irish police (the gardai) seeking the extradition from Scotland of two men in connection with Donaldson's murder who are not connected to either the now-disbanded Provisional IRA nor Sinn Féin. The article further stated: \"Gardai say they have no evidence to support claims that the killing was sanctioned by Adams.\" It quoted a garda source as saying: \"We have no evidence to put to Gerry Adams. The investigation team would have no reason to invite him in for questioning or even attempt to arrest him... we have built up a detailed picture of those involved in Donaldson's murder. Adams does not feature.\" So where does that leave Spotlight's exclusive? I sent these questions to the BBC Northern Ireland press office earlier today: Given that the \"evidence\" of the source known as \"Martin\" has been rubbished by Adams, his solicitor, Donaldson's family, and the gardai, does the BBC still stand by the claims he made? On reflection, do the people responsible for the programme really think Martin's allegations were credible? What steps were taken to ensure that he wasn't a fantasist? Is the BBC happy with relying on a single source making controversial allegations, which appears to contravene the corporation's guidelines? Was the documentary run past the BBC's editorial and standards department prior to transmission? Finally, given that a Spotlight in 2014 levelled contested claims at Adams, does it not worry the BBC that it could be accused of attempting to smear the Sinn Fein leader? In response, a BBC spokesperson issued a statement saying: \"The programme dealt with matters of great public interest and fully met the BBC's editorial standards. The BBC stands by its journalism.\" NB: I have previously written for Sinn Féin's newspaper, An Phoblacht.",
    "rewritten": """Has BBC's Spotlight Gone Too Far? Questions Arise Over Adams Murder Allegation

Spotlight, BBC Northern Ireland's acclaimed current affairs program with a 43-year history of awards, seems to have strayed, in my opinion, after its recent broadcast concerning the 2006 killing of Sinn Féin official Denis Donaldson. Donaldson's death occurred shortly after he admitted to being a British agent for nearly two decades. The program's central assertion was that Sinn Féin leader Gerry Adams had ordered the murder. Given Spotlight's standing, this allegation made headlines in Ireland's leading newspapers and even some in the UK.

The program, titled "Spy in the IRA," relied on the "evidence" of a single, anonymous source. His identity and voice were disguised "because of fears for his safety." He was presented as a former IRA volunteer and Sinn Féin member who himself had become an informer for the British secret services. Spotlight reporter Jennifer O'Leary hyped his "exclusive interview" in a pre-broadcast preview article, stating, "For months, I've been in contact with a man, referred to in the programme as 'Martin'." However, his on-screen statements lacked any semblance of truthfulness, amounting to staccato answers to leading questions. There wasn't a shred of supporting evidence for his claims that Adams, a member of the Irish parliament, had sanctioned the murder.

As a journalistic endeavor, the program appeared entirely inadequate. It didn't take long for the allegation about Adams's involvement to fall apart. This wasn't particularly surprising, as the Real IRA—one of the dissident groups opposed to Sinn Féin—had claimed responsibility for Donaldson's killing about three years after his death.

First, Adams issued a statement, reported on the BBC website, unequivocally refuting "these unsubstantiated allegations," which he viewed as an attempt to smear him. Second, Adams's solicitor followed up with another clear rebuttal, reported by The Guardian, stating his client had "no knowledge of and... no involvement whatsoever in the killing of Denis Donaldson." Third, The Guardian reported on Friday that Donaldson's relatives had rejected the Spotlight claim, with their solicitor calling it "absolute nonsense." Fourth, and most conclusively, the Irish edition of The Sunday Times ran a story about the Irish police (the Gardaí) seeking the extradition from Scotland of two men in connection with Donaldson's murder. These individuals are not connected to the now-disbanded Provisional IRA or Sinn Féin. The article further stated: "Gardai say they have no evidence to support claims that the killing was sanctioned by Adams." It quoted a Garda source as saying, "We have no evidence to put to Gerry Adams. The investigation team would have no reason to invite him in for questioning or even attempt to arrest him... we have built up a detailed picture of those involved in Donaldson's murder. Adams does not feature."

So, where does that leave Spotlight's exclusive? Earlier today, I sent these questions to the BBC Northern Ireland press office:

Given that the "evidence" from the source known as "Martin" has been discredited by Adams, his solicitor, Donaldson's family, and the Gardaí, does the BBC still stand by his claims?

Upon reflection, do those responsible for the program truly believe Martin's allegations were credible?

What steps were taken to ensure he wasn't a fantasist?

Is the BBC comfortable relying on a single source making controversial allegations, which seems to contravene the corporation's guidelines?

Was the documentary reviewed by the BBC's editorial and standards department before transmission?

Finally, considering that a 2014 Spotlight program also leveled contested claims against Adams, does the BBC not worry about being accused of attempting to smear the Sinn Féin leader?

In response, a BBC spokesperson issued a statement saying: "The programme dealt with matters of great public interest and fully met the BBC's editorial standards. The BBC stands by its journalism."

NB: I have previously contributed to Sinn Féin's newspaper, An Phoblacht."""
}
data.append(entry)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Appended 1 entry to {output_path}")