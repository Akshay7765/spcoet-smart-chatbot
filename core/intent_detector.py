# core/intent_detector.py
# ──────────────────────────────────────────────────────────────
#  Intent Detection + Confidence Scoring + Suggestions
#  Uses keyword matching with weighted scoring
#  Detects: admission, fees, courses, hostel, placement,
#           faculty, contact, events, facilities, general
# ──────────────────────────────────────────────────────────────

import re
from typing import Tuple


class IntentDetector:
    """
    Detects user intent from normalized text.
    Returns intent name, confidence score (0-1), sub-intent, and suggestions.
    """

    # ── Intent Patterns ───────────────────────────────────────
    # Format: { intent_name: { weight: int, keywords: [...], sub_intents: {...} } }
    INTENTS = {
        "admission": {
            "weight"  : 1.0,
            "keywords": [
                "admission","admit","apply","application","enroll","enrollment",
                "join","get into","seat","cap","round","mht","cet","jee","neet",
                "cut","cutoff","merit","wait","waitlist","open","close","deadline",
                "date","last date","form","fill","registration","register",
                "intake","document","eligibility","eligible","qualify","qualify"
            ],
            "sub_intents": {
                "eligibility": ["eligible","eligibility","qualify","criteria","requirement","minimum","percentage","marks"],
                "documents"  : ["document","certificate","marksheet","aadhaar","caste","income","domicile"],
                "dates"      : ["date","deadline","last date","when","schedule","calendar"],
                "process"    : ["how","process","step","procedure","apply","fill"],
                "management_quota": ["management","quota","direct","institute","level"]
            }
        },
        "fees": {
            "weight"  : 1.0,
            "keywords": [
                "fee","fees","tuition","cost","amount","pay","payment","expensive",
                "scholarship","financial","aid","installment","per year","annual",
                "semester","hostel fee","mess","total cost","fra","structure","waiver"
            ],
            "sub_intents": {
                "scholarship": ["scholarship","free","waiver","help","financial","aid","ebc","obc","sc","st"],
                "hostel_fee" : ["hostel fee","room","mess","accomm"],
                "total"      : ["total","all","complete","full","everything"]
            }
        },
        "courses": {
            "weight"  : 0.9,
            "keywords": [
                "course","courses","branch","branches","department","engineering",
                "computer","mechanical","civil","electrical","mba","program","degree",
                "be","btech","bscit","mtech","me","subjects","syllabus","specialization",
                "scope","career","job","after","stream","offer","available","intake","seat"
            ],
            "sub_intents": {
                "computer"   : ["computer","cs","cse","it","software","coding","programming"],
                "mechanical" : ["mechanical","mech","machine","automobile"],
                "civil"      : ["civil","construction","structure","design"],
                "electrical" : ["electrical","elec","power","electronics"],
                "mba"        : ["mba","management","business","administration"]
            }
        },
        "hostel": {
            "weight"  : 0.9,
            "keywords": [
                "hostel","accommodation","stay","room","dormitory","pg","boarding",
                "mess","food","warden","girls","boys","separate","facility","wifi",
                "cctv","security","laundry","in-campus","on campus"
            ],
            "sub_intents": {
                "girls"      : ["girls","women","female","ladies"],
                "boys"       : ["boys","men","male","gents"],
                "food"       : ["food","meal","mess","breakfast","lunch","dinner","canteen"],
                "facility"   : ["wifi","internet","ac","gym","laundry","facility"]
            }
        },
        "placement": {
            "weight"  : 1.0,
            "keywords": [
                "placement","placements","job","jobs","hire","hiring","recruit",
                "company","companies","tcs","infosys","wipro","package","salary","lpa",
                "campus","drive","tnp","t&p","tp","training","percentage","record",
                "average","highest","past","previous","year"
            ],
            "sub_intents": {
                "companies"  : ["company","companies","recruiter","which","tcs","infosys","wipro","bosch"],
                "package"    : ["package","salary","lpa","ctc","highest","average","earning"],
                "record"     : ["record","history","previous","last year","how many","percentage"]
            }
        },
        "faculty": {
            "weight"  : 0.8,
            "keywords": [
                "faculty","teacher","professor","staff","hod","head","department head",
                "principal","dean","phd","qualification","experience","lecturer",
                "guide","mentor","research","paper","publication"
            ],
            "sub_intents": {
                "hod"        : ["hod","head of department","head","incharge"],
                "principal"  : ["principal","director","head of institution"],
                "count"      : ["how many","number","count","total","strength"]
            }
        },
        "facilities": {
            "weight"  : 0.8,
            "keywords": [
                "facilit","lab","laboratory","library","sport","sports","gym","gymnasium",
                "ground","playground","auditorium","seminar","conference","wifi","internet",
                "canteen","cafe","bus","transport","vehicle","shuttle","nss","club",
                "activity","event","tech","fest","cultural","computer lab","smart class",
                "atm","medical","dispensary","shop"
            ],
            "sub_intents": {
                "library"    : ["library","book","journal","digital","resource"],
                "sports"     : ["sport","cricket","football","volleyball","basketball","indoor"],
                "transport"  : ["bus","transport","route","shuttle","travel","commute"],
                "labs"       : ["lab","laboratory","computer","equipment","machine"]
            }
        },
        "contact": {
            "weight"  : 0.9,
            "keywords": [
                "contact","phone","number","call","email","address","location","where",
                "website","reach","find","map","direction","how to go","near",
                "office","timing","working hour","open","close"
            ],
            "sub_intents": {
                "phone"   : ["phone","number","call","mobile","helpline"],
                "address" : ["address","location","where","direction","how to reach","map"],
                "email"   : ["email","mail","gmail","send","write"],
                "website" : ["website","site","url","online","portal","web"]
            }
        },
        "events": {
            "weight"  : 0.7,
            "keywords": [
                "event","events","fest","festival","technothon","somotsav","cultural",
                "technical","hackathon","competition","contest","seminar","workshop",
                "webinar","conference","expo","notice","announcement","news","update",
                "nss","camp","blood","donation","activity","program","program"
            ],
            "sub_intents": {
                "cultural"  : ["cultural","somotsav","dance","music","drama"],
                "technical" : ["technical","technothon","coding","hackathon","tech"],
                "nss"       : ["nss","camp","service","blood","donation"]
            }
        },
        "about": {
            "weight"  : 0.7,
            "keywords": [
                "about","college","spcoet","history","established","founded","when",
                "who","society","trust","aicte","sppu","university","naac","nba",
                "accredit","affiliated","recognition","vision","mission","overview",
                "principal message","secretary","president","trustee","management"
            ],
            "sub_intents": {
                "history"    : ["history","when","established","founded","year"],
                "accreditation":["naac","nba","aicte","accredit","affiliated","approved"],
                "leadership" : ["principal","secretary","president","trustee","management"]
            }
        },
        "general": {
            "weight"  : 0.0,
            "keywords": ["hi","hello","hey","help","thanks","thank","bye","good","ok","yes","no"],
            "sub_intents": {}
        }
    }

    # Suggestions per intent (shown as quick-reply chips)
    SUGGESTIONS = {
        "admission" : ["What documents are needed?", "What is the fee structure?", "When does admission start?"],
        "fees"      : ["Are there scholarships?", "What is the hostel fee?", "How to pay fees?"],
        "courses"   : ["What is the intake for Computer Engineering?", "Tell me about MBA program", "Career scope after BE?"],
        "hostel"    : ["Hostel fee details?", "Is girls hostel available?", "What facilities are in hostel?"],
        "placement" : ["Which companies recruit from SPCOET?", "What is the highest package?", "How is T&P training done?"],
        "faculty"   : ["Who is the HOD of CS department?", "What is the faculty qualification?", "How many professors?"],
        "facilities": ["Tell me about the library", "Sports facilities?", "Bus routes available?"],
        "contact"   : ["What is the college address?", "What are office timings?", "College website?"],
        "events"    : ["What is TECHNOTHON?", "Tell me about NSS activities", "Upcoming events?"],
        "about"     : ["NAAC accreditation details?", "When was SPCOET established?", "What is the vision of college?"],
        "general"   : ["Tell me about courses", "Admission process?", "Contact information?"]
    }

    def detect(self, text: str) -> dict:
        """
        Detect intent from normalized text.
        Returns: { intent, sub_intent, confidence, all_scores }
        """
        if not text or not text.strip():
            return {"intent": "general", "sub_intent": "", "confidence": 0.5}

        text_lower = text.lower()
        scores     = {}

        # Score each intent
        for intent_name, intent_data in self.INTENTS.items():
            score = self._score_intent(text_lower, intent_data["keywords"])
            scores[intent_name] = score * intent_data["weight"]

        # Find best intent
        best_intent = max(scores, key=scores.get)
        best_score  = scores[best_intent]

        # Normalize confidence to 0–1
        total      = sum(scores.values())
        confidence = (best_score / total) if total > 0 else 0.5

        # Boost confidence if score is very high
        if best_score >= 5:
            confidence = min(confidence * 1.5, 0.99)
        elif best_score == 0:
            best_intent = "general"
            confidence  = 0.5

        # Detect sub-intent
        sub_intent = ""
        if best_intent in self.INTENTS:
            sub_intent = self._detect_sub_intent(text_lower, self.INTENTS[best_intent]["sub_intents"])

        return {
            "intent"     : best_intent,
            "sub_intent" : sub_intent,
            "confidence" : min(round(confidence, 2), 0.99),
            "all_scores" : {k: round(v, 2) for k, v in sorted(scores.items(), key=lambda x: -x[1])[:5]}
        }

    def _score_intent(self, text: str, keywords: list) -> float:
        """Count keyword matches (with partial match support)."""
        score = 0.0
        for kw in keywords:
            if kw in text:
                # Exact word match scores higher
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    score += 1.5
                else:
                    score += 0.5
        return score

    def _detect_sub_intent(self, text: str, sub_intents: dict) -> str:
        """Detect sub-intent within the primary intent."""
        best_sub   = ""
        best_score = 0
        for sub_name, keywords in sub_intents.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_sub   = sub_name
        return best_sub

    def get_suggestions(self, intent: str) -> list:
        """Return 3 follow-up suggestion chips for the detected intent."""
        return self.SUGGESTIONS.get(intent, self.SUGGESTIONS["general"])

    def get_did_you_mean(self, text: str, intent: str) -> str:
        """Generate 'Did you mean...?' suggestion for ambiguous queries."""
        if intent == "general" and len(text.split()) <= 2:
            common = {
                "fee"      : "Did you mean: What is the fee structure?",
                "course"   : "Did you mean: What courses are offered?",
                "hostel"   : "Did you mean: What hostel facilities are available?",
                "place"    : "Did you mean: Placement information?",
                "admit"    : "Did you mean: How to get admission?",
                "contact"  : "Did you mean: How to contact SPCOET?"
            }
            for kw, msg in common.items():
                if kw in text.lower():
                    return msg
        return ""
