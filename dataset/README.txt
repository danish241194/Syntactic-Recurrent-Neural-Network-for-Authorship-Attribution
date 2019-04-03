Problem A: traditional authorship attribution, 3 authors, closed class.
Problem B: as above (and same training data), open class
Problem C: traditional authorship attribution, 8 authors, closed class
Problem D: as above (and same training data), open class
Problem E: mixed paragraph documents
Problem F: intrusive paragraph documents
Problem G: closed-class
Problem H: open-class

Training documents are named using the following scheme:

12XtrainZ03 would be the 03rd training document for the 26th author
(Z) in problem X (in the PAN-12 corpus).   There would be a
corresponding set of documents named 12Xtest03 -- this would be the
3rd test document for problem X.  If this were an authorship
attribution problem, then the participants would be expected to return
the identifying letter of the putative author (e.g., Z).

For problems E and F, it's a little different as there is no training
data.  Instead I provide "sample" documents 12Esample01 and
12Fsample01.

12Esample01 is six paragraphs long.   Odd paragraphs are from "A
Princess of Mars"; even are from "The Adventures of Sherlock Holmes."
 Participants would be expected to return the two sets (1,3,5) and
(2,4,6) in some appropriate notation.

Similarly, 12Fsample01 is mostly from "Peter Pan," but paragraphs 7-8
are interlopers from "The Communist Manifesto."   Participants would
be expected to return (7,8).

Perhaps obviously, these sample documents are particularly ill-suited
for competition and the actual documents will be more difficult.  (How
hard is it to distinguish political theory from children's fiction?)
I have also not formalized the format in which responses are to be
provided.   I am open to discussion on this point prior to release or
even for a bit once people look at this.

Problems G (closed-class) and H (open-class)
are novel-length analogues to problems A/B and problems C/D; the
training data represent novel-length samples of works by the authors
named in the file names.   These authors (as well as possibly others
in the case of problem H) will be represented by novels in the
training data.
