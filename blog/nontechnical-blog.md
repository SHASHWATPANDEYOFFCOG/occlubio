# Finding One Face in a Packed Stadium

Imagine you're at a sold-out stadium — 80,000 people — and a friend says, "My cousin
is here somewhere. Find her." You've seen one photo of the cousin, once. You could
walk the stands row by row, squinting at every face. It would take days, and by seat
40,000 you'd be second-guessing yourself constantly: *wasn't that her? No… close,
though.*

Now imagine doing it in the blink of an eye, without getting worse at it as the
stadium fills up. That — finding one specific face among enormous numbers of people,
quickly and reliably — is the problem our system is built to solve.

## Checking an ID vs. scanning the whole building

Most people have met face recognition through their phone: it looks at you, decides
"yes, this is the owner," and unlocks. That's a one-to-one check — one face against
one stored identity. It's like a doorman glancing at your ID card: easy, because the
question is only *"are you who you say you are?"*

Our system answers a much harder question: *"who is this, out of everyone we know?"*
No name is offered up front. A face arrives, and the system must compare it against
**every registered person at once** — one-to-many. That's not the doorman checking
your card; that's the doorman recognizing any of 100,000 members on sight, instantly,
including telling strangers apart from members who just got a haircut.

## Every face becomes a numeric fingerprint

Computers don't see faces the way we do. So the first step is translation: software
finds the face in the photo, straightens it so the eyes and mouth sit in standard
positions, and then converts it into a list of 512 numbers — think of it as a
**numeric fingerprint** of that face.

The magic property of these fingerprints: two photos of the *same* person — different
day, different lighting, different haircut — produce lists of numbers that are very
*similar*. Photos of *different* people produce numbers that are clearly *different*.

[Simple diagram: how a face becomes a match — photo → straightened face →
numeric fingerprint → compared against the library of registered fingerprints]

Registering someone means storing their numeric fingerprint in a library, next to
their name. Recognizing someone means taking a new photo, computing its fingerprint,
and asking the library: *whose stored fingerprint is closest to this one?* If the
closest match is close *enough*, the system says "this is Maya." If nothing is close
enough, it honestly answers "I don't know this person" — which matters just as much
as getting matches right.

## Why "it works in the demo" isn't good enough

Here's the part most people never think about: a system like this can work
beautifully for 100 registered people and quietly fall apart at a million.

Two things go wrong as the library grows. First, **speed**: more fingerprints means
more comparisons, and the search that felt instant with 100 people can slow to a
crawl. Second — and sneakier — **mistakes**: the more people you register, the more
likely it becomes that *someone* in the library happens to look a bit like the
stranger in front of the camera. Every new registration is another chance for a
coincidental near-twin. A packed stadium doesn't just take longer to search than an
empty one; it contains more lookalikes.

## The dress rehearsal

So we didn't just build the system — we stress-tested it the way you'd stress-test a
bridge: by loading it far past comfortable and measuring exactly when and how it
strains.

We used a public research collection of celebrity photos with about **10,000
different people** in it. We registered them, then tested with *different* photos of
the same people (can it still find them?) and with photos of people deliberately
left out of the library (does it correctly say "I don't know you," or does it
falsely accuse a stranger of being someone else?). Then, to see how the search
performs at truly large scale, we grew the library to **half a million entries**
and re-measured everything: how fast the slowest searches are, not just the average;
how much memory the library eats; how accuracy shifts as the crowd grows.

We also compared two search styles: a **meticulous librarian** who checks every
single fingerprint (always right, but slower as shelves fill up) and a **clever
librarian** who organizes the shelves so she only checks the most promising sections
(dramatically faster, very rarely misses). Knowing exactly what "very rarely" means,
and when the clever approach is worth it — that was the point of the test.

## Where this is useful — and what it isn't

Systems like this power familiar, mostly boring conveniences: unlocking devices,
badge-free entry to an office, finding a lost person's appearances across hours of
video, de-duplicating identity records.

But two honest notes belong in any fair description:

**It is not flawless.** Accuracy is a dial, not a guarantee. Set the matching bar
too low and strangers get waved through; too high and legitimate people get rejected.
Good systems publish those error rates, test them at realistic scale — including
across different faces, lighting, and image quality — and keep a human in the loop
for consequential decisions.

**A face is not a password.** You can't change your face if it leaks. Those numeric
fingerprints are biometric data and deserve the same care as medical records:
collected with consent, stored encrypted, kept only as long as needed, and never
quietly repurposed. The engineering question "can we find one face among a million?"
always travels with the human question "*should* we, here, and with whose
permission?" A trustworthy system treats both as part of the design.

## The short version

Turn every face into a numeric fingerprint. Keep a library of them. When a new face
appears, find the closest fingerprint — fast, even when the library holds half a
million entries — and be willing to say "no match" when that's the truth. Then test
the whole thing at stadium scale *before* the stadium shows up, so you know it stays
quick and careful when the crowd arrives.
