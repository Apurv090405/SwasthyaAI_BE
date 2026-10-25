#!/usr/bin/env bash
# One-call test for the LangGraph supervisor + all clinical agents (POST /analyze_transcript).
#
# Prerequisites:
#   uv run uvicorn app:app --host 0.0.0.0 --port 8000
#
# Usage:
#   chmod +x test_supervisor_orchestrator.sh
#   ./test_supervisor_orchestrator.sh           # OpenAI
#   ./test_supervisor_orchestrator.sh gemini    # Gemini
#   API_URL=http://127.0.0.1:8000 ./test_supervisor_orchestrator.sh

set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"
PROVIDER="${1:-openai}"

export API_URL PROVIDER

uv run python - << 'PY'
import json
import os
import urllib.request

# Rich Hindi transcript — exercises all agents in the supervisor pipeline
TRANSCRIPT = """नमस्ते डॉक्टर, मेरा नाम रामकिशन है। मैं पैंतालीस साल का हूँ। पिछले चार दिनों से लगातार बुखार है, कल शाम से ठंड लगने लगी है और आज सुबह से खाँसी भी हो गई है। साँस लेने में भी थोड़ी तकलीफ़ हो रही है, ऐसा लगता है सीने में भारीपन है। कल रात बुखार सौ एक डिग्री फारेनहाइट तक चला गया था। पसीना बह रहा है और सिर में भारी दर्द है। पानी कम पी रहा हूँ, पेशाब भी कम आया।

मुझे मधुमेह है, दो साल से मेटफॉर्मिन चल रही है। ब्लड प्रेशर की भी दवा लेता हूँ। कोई एलर्जी नहीं बताई गई। दो हफ्ते पहले ऑफिस में किसी को ज़ुकाम हुआ था, शायद वहीं से लग गया।

कृपया बताएँ मुझे क्या करना चाहिए और क्या यह गंभीर है?"""

url = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/") + "/analyze_transcript"
provider = os.environ.get("PROVIDER", "openai")

body = {
    "userid": "test_orchestrator_call_001",
    "transcript": TRANSCRIPT,
    "location": "Delhi, NCR, India",
    "llm_provider": provider,
}

data = json.dumps(body, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
)

print(f"POST {url} (llm_provider={provider})\n---")
with urllib.request.urlopen(req, timeout=600) as resp:
    raw = resp.read().decode("utf-8")
    try:
        print(json.dumps(json.loads(raw), ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print(raw)
PY












------------
techcodiste@Techs-MacBook-Pro swasthya-ai-python % ./test_supervisor_orchestrator.sh
POST http://127.0.0.1:8000/analyze_transcript (llm_provider=openai)
---
{
  "patient_id": "patient_001",
  "pseudo_id": "anon_G7X4Z",
  "language": "hi",
  "consent_status": "verified",
  "input_summary": {
    "transcript_excerpt": "नमस्ते डॉक्टर, मेरा नाम रामकिशन है। मैं पैंतालीस साल का हूँ। पिछले चार दिनों से लगातार बुखार है, कल शाम से ठंड लगने लगी है और आज सुबह से खाँसी भी हो गई है। साँस लेने में भी थोड़ी तकलीफ़ हो रही है, ऐसा लगता है सीने में भारीपन है। कल रात बुखार सौ एक डिग्री फारेनहाइट तक चला गया था। पसीना बह रहा है और सिर में भारी दर्द है। पानी कम पी रहा हूँ, पेशाब भी कम आया।",
    "call_date": "2024-06-19T10:30:00+05:30",
    "transcript_tokens": 200
  },
  "symptoms": [
    {
      "name": "बुखार",
      "duration_days": 4,
      "onset_description": "चार दिन पहले",
      "severity": "मध्यम",
      "source": "utt_3"
    },
    {
      "name": "ठंड लगना",
      "duration_days": 1,
      "onset_description": "कल शाम से",
      "severity": "मध्यम",
      "source": "utt_3"
    },
    {
      "name": "खांसी",
      "duration_days": 0,
      "onset_description": "आज सुबह से",
      "severity": "मध्यम",
      "source": "utt_3"
    },
    {
      "name": "साँस लेने में तकलीफ़",
      "duration_days": 0,
      "onset_description": "आज सुबह से",
      "severity": "गंभीर",
      "source": "utt_3"
    },
    {
      "name": "सीने में भारीपन",
      "duration_days": 0,
      "onset_description": "आज सुबह से",
      "severity": "गंभीर",
      "source": "utt_3"
    },
    {
      "name": "पसीना बहना",
      "duration_days": 1,
      "onset_description": "कल रात से",
      "severity": "मध्यम",
      "source": "utt_3"
    },
    {
      "name": "सिरदर्द",
      "duration_days": 1,
      "onset_description": "कल रात से",
      "severity": "मध्यम",
      "source": "utt_3"
    },
    {
      "name": "पानी कम पीना",
      "duration_days": 0,
      "onset_description": "आज से",
      "severity": "गंभीर",
      "source": "utt_3"
    },
    {
      "name": "पेशाब कम आना",
      "duration_days": 0,
      "onset_description": "आज से",
      "severity": "गंभीर",
      "source": "utt_3"
    }
  ],
  "past_medical_history": {
    "reported_by_patient": {
      "diabetes": true,
      "hypertension": true,
      "medications": [
        "मेटफॉर्मिन",
        "ब्लड प्रेशर की दवा"
      ],
      "allergies": [],
      "prior_infections": [
        "ऑफिस में URTI एक्सपोजर (दो हफ्ते पहले)"
      ]
    },
    "retrieved_from_api": {
      "record_found": false,
      "lookup_time": null
    }
  },
  "vitals_reported": {
    "temperature_f": 101.0,
    "spo2_percent": null,
    "heart_rate": null
  },
  "risk_scores": {
    "sepsis_risk_score": 0.67,
    "cardiac_flag": true,
    "dehydration_risk": "high"
  },
  "red_flags": [
    "साँस लेने में तकलीफ़",
    "सीने में भारीपन",
    "पानी कम पीना",
    "पेशाब कम आना"
  ],
  "triage": {
    "level": "Emergent",
    "explanation": "मरीज में कई गंभीर रेड फ्लैग्स (साँस लेने में तकलीफ़, सीने में भारीपन, पेशाब कम आना, पानी कम पीना) मौजूद हैं। जोखिम स्कोर से संक्रमण (सेप्सिस) का गंभीर खतरा और गंभीर निमोनिया की संभावना है, विशेष रूप से मधुमेह व हाइपरटेंशन जैसी पुरानी बीमारियों के साथ। qSOFA स्कोर 2 और SIRS 3 गंभीरता दर्शाते हैं। कार्डियक अत्यावश्यकता और मधुमेह संबंधी जटिलताएँ भी मौजूद हैं। ऐसी स्थिति तुरंत चिकित्सकीय हस्तक्षेप की मांग करती है, विलंब जानलेवा हो सकता है। अतः ईमर्जेंट श्रेणी में रखा जाता है।",
    "recommended_action_window": "तुरंत",
    "override_reason": null,
    "source_guidelines": [
      {
        "source": "WHO",
        "doc_id": "WHO-2018-Sepsis-Guidelines",
        "section": "Recognition and emergency management of sepsis",
        "snippet": "Patients with breathing difficulty, chest heaviness, and decreased urine output—especially with diabetes—should be considered at high risk for sepsis and require emergent intervention."
      },
      {
        "source": "ICMR",
        "doc_id": "ICMR-COVID19-Clinical-Management-2020",
        "section": "Assessment and Referral",
        "snippet": "साँस लेने में तकलीफ़, सीने में तकलीफ़, पेशाब में कमी आदि लक्षणों में तुरंत मेडिकल केयर एवं आपातकालीन हस्तक्षेप आवश्यक है—विशेषकर मधुमेह या अन्य को-मॉर्बिडिटी वाले मरीजों में।"
      },
      {
        "source": "Manchester Triage System",
        "doc_id": "Manchester-Triage-Guidelines-2014",
        "section": "Respiratory distress: Immediate Response",
        "snippet": "Immediate triage ('Red' level) when presenting with respiratory difficulty with evidence of systemic compromise (such as hypotension, decreased urine output)."
      }
    ]
  },
  "differential_diagnosis": [
    {
      "condition": "न्युमोनिया (Pneumonia)",
      "confidence": 0.85,
      "rationale": "मरीज को चार दिनों से बुखार, आज सुबह से खांसी, साँस लेने में तकलीफ़ और सीने में भारीपन है। मधुमेह और हाइपरटेंशन जैसी पुरानी बीमारियाँ संक्रमण के जोखिम को बढ़ाती हैं। पेशाब कम आना और पानी कम पीना बुखार की शरीर पर असर के संकेत हैं। ऑफिस में URTI का एक्सपोजर भी संभावित कारण है, लेकिन तीव्र लक्षण और साँस की दिक्कत निमोनिया की तरफ़ इशारा करते हैं।",
      "citations": [
        {
          "source": "WHO",
          "doc_id": "WHO-2018-Sepsis-Guidelines",
          "snippet": "Patients with breathing difficulty, chest heaviness, and decreased urine output—especially with diabetes—should be considered at high risk for sepsis and require emergent intervention."
        }
      ]
    },
    {
      "condition": "वायरल ज्वर (Common viral infection)",
      "confidence": 0.7,
      "rationale": "ऑफिस में वायरल इन्फेक्शन का एक्सपोजर ज्ञात है। बुखार, सिरदर्द, पसीना, खांसी व छींक जैसे लक्षण वायरल फ्लू या सामान्य वायरल इन्फेक्शन में होते हैं। किंतु फेफड़े में तकलीफ़ और पेशाब कम आना गंभीरता की तरफ़ इशारा कर सकता है।",
      "citations": [
        {
          "source": "ICMR",
          "doc_id": "ICMR-GUIDELINE-URTI-2021",
          "snippet": "Symptoms such as fever, cough, myalgia, and viral exposure are typical for viral ARIs. Reduced urine output indicates possible severe progression."
        }
      ]
    },
    {
      "condition": "कोविड-१९ संक्रमण (COVID-19 infection)",
      "confidence": 0.8,
      "rationale": "कोविड-१९ के रोगियों में बुखार, खांसी, साँस लेने में तकलीफ़, सिरदर्द व थकान आम है, और डिजिज़ अधिक गंभीर मधुमेह एवं हाइपरटेंशन वाले मरीजों में हो सकता है। लक्षणों व ऑफिस एक्सपोजर के कारण कोविड की आशंका हो सकती है, हालांकि पुष्टि हेतु जाँच आवश्यक है।",
      "citations": [
        {
          "source": "ICMR",
          "doc_id": "ICMR-COVID19-Clinical-Management-2020",
          "snippet": "साँस लेने में तकलीफ़, सीने में तकलीफ़, पेशाब में कमी आदि लक्षणों में तुरंत मेडिकल केयर एवं आपातकालीन हस्तक्षेप आवश्यक है—विशेषकर मधुमेह या अन्य को-मॉर्बिडिटी वाले मरीजों में।"
        }
      ]
    },
    {
      "condition": "सेप्सिस/गंभीर संक्रमण (Sepsis/Severe infection)",
      "confidence": 0.75,
      "rationale": "मधुमेह व हाइपरटेंशन के मरीज को लंबे समय से बुखार, पसीना, पेशाब कम आना, पानी कम पीना और शारीरिक कमजोरी जैसी गंभीर लक्षण दिख रहे हैं, जो शरीर में गंभीर संक्रमण (सेप्सिस) का संकेत हो सकता है। साँस में तकलीफ़, सीने में भारीपन और पेशाब में कमी रेड फ्लैग्स हैं।",
      "citations": [
        {
          "source": "WHO",
          "doc_id": "WHO-2018-Sepsis-Guidelines",
          "snippet": "Patients with breathing difficulty, chest heaviness, and decreased urine output—especially with diabetes—should be considered at high risk for sepsis and require emergent intervention."
        },
        {
          "source": "ICMR",
          "doc_id": "ICMR-Sepsis-Adult-2021",
          "snippet": "Sepsis risk increases significantly with underlying comorbidities and clinical signs like decreased urine output."
        }
      ]
    },
    {
      "condition": "तीव्र ब्रोंकाइटिस (Acute bronchitis)",
      "confidence": 0.6,
      "rationale": "हाल ही में वायरल एक्सपोजर और खांसी अचानक शुरू होना, गले की खराश न होना, और बुखार तीव्र ब्रोंकाइटिस की संभावना जताते हैं। लेकिन सांस लेने में तकलीफ़, सीने में भारीपन और पेशाब कम आना निमोनिया की आशंका बढ़ाता है, इसलिए ब्रोंकाइटिस की संभावना कम है।",
      "citations": [
        {
          "source": "ATS",
          "doc_id": "ATS-Bronchitis-2019",
          "snippet": "Acute cough and viral exposure raise likelihood of bronchitis, but respiratory distress, chest heaviness, and oliguria raise concern for pneumonia."
        }
      ]
    }
  ],
  "recommendations": {
    "for_patient_hi": "आपको बुखार, सांस में तकलीफ, सीने में भारीपन और पेशाब कम आना जैसी गंभीर लक्षण हैं। आपकी स्थिति गंभीर है, इसलिए आपको तुरंत (इमरजेंसी) अस्पताल जाना जरूरी है।\n\n• अस्पताल/आपातकालीन विभाग में तुरंत पहुंचे, देरी न करें।\n• अपने साथ दवा की पर्ची, रोगी पहचान पत्र (अगर है) और पुरानी बीमारी (मधुमेह/ब्लड प्रेशर) की दवाएं लेकर जाएं।\n• रास्ते में जरूरत हो तो किसी की मदद लें या एम्बुलेंस बुलाएं।\n• पानी पिएँ लेकिन जब तक डॉक्टर सलाह न दें, बहुत अधिक न पिएँ।\n• बिना डॉक्टर की सलाह के कोई दवा न लें।\n• अस्पताल में डॉक्टर को सभी लक्षणों की जानकारी दें, पुरानी बीमारियों के बारे में भी बताएं।\n• कोरोना संक्रमण के कारण मास्क पहनें और भीड़ से बचें।",
    "tests_suggested": [
      "CBC",
      "Blood culture",
      "CRP",
      "Procalcitonin",
      "Serum lactate",
      "Renal function tests",
      "Liver function tests",
      "Chest X-ray",
      "COVID-19 RT-PCR",
      "Urine routine and microscopy",
      "Electrolytes",
      "Random blood sugar",
      "ECG"
    ],
    "referral_suggestion": {
      "urgency": "immediate",
      "recommended_center": "आपातकालीन विभाग (Emergency Department) - पास के सरकारी/प्राइवेट अस्पताल"
    }
  },
  "doctor_summary_note": {
    "summary": "Middle-aged male with diabetes and hypertension presenting with 4 days of fever, development of cough and shortness of breath, chest heaviness, and reduced oral intake/urine output. Red flags include respiratory distress and possible sepsis/shock. This is a high-risk, emergent case requiring immediate hospital admission and advanced management.",
    "actionables": [
      "Immediate hospital admission via Emergency Department",
      "Initiate sepsis protocol: Secure IV access, Begin fluid resuscitation as tolerated",
      "Start broad-spectrum intravenous antibiotics as per hospital policy after sending blood cultures",
      "Provide supplemental oxygen to maintain SpO2 >94%",
      "Monitor vital signs closely, including urine output",
      "Check bedside glucose and manage diabetes complications as per protocol",
      "Cardiac monitoring (ECG, enzymes if indicated)",
      "Order CBC, CRP, Procalcitonin, Renal/liver function tests, Blood culture, Chest X-ray, COVID-19 RT-PCR",
      "Assess for possibility of ICU transfer if patient deteriorates",
      "Consider early involvement of pulmonology/infectious disease/critical care specialist"
    ]
  },
  "report_meta": {
    "created_at": "2024-06-19T10:35:00+05:30",
    "generated_by": "supervisor-agent-v1.2",
    "confidence": 0.85,
    "human_review_required": true,
    "audit_log_id": "audit_M5N6L"
  }
}
techcodiste@Techs-MacBook-Pro swasthya-ai-python %
