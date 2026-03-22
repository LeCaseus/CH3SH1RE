from backend.router import detect_intent

tests = [
    "can you rewrite my cv",
    "what is the weather today",
    "help me plan a trip to Auckland",
    "I am feeling really stressed lately",
    "what is photosynthesis",
    "compare iPhone vs Samsung",
]

for t in tests:
    fn = detect_intent(t)
    print(f"'{t}' → {fn.__name__}")
