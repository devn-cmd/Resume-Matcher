import json
from pathlib import Path
from typing import Dict, List, Any, Set

class SkillRecommender:
    def __init__(self, trends_path: str = "data/market_trends.json"):
        self.trends_path = Path(trends_path)
        self.market_trends = self._load_trends()

    def _load_trends(self) -> Dict[str, Any]:
        """Loads static market trends from JSON."""
        if not self.trends_path.exists():
            return {}
        with open(self.trends_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_roadmap(self, missing_skills: Set[str], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Ranks missing skills by market demand. If a skill doesn't have a 
        custom roadmap, it generates a smart generic fallback.
        """
        recommendations = []
        
        for skill in missing_skills:
            skill_lower = skill.lower()
            
            # Check if we have explicit roadmap data for this skill
            if skill_lower in self.market_trends:
                trend_data = self.market_trends[skill_lower]
                recommendations.append({
                    "skill": skill,
                    "demand_score": trend_data.get("demand_score", 0.0),
                    "roadmap": trend_data.get("roadmap", [])
                })
            else:
                # CHOICE B: Fallback logic for recognized skills without an explicit roadmap
                recommendations.append({
                    "skill": skill,
                    "demand_score": 0.30,  # Lower baseline priority so explicit trends rank higher
                    "roadmap": [
                        {
                            "step": 1, 
                            "action": f"Review the official documentation and getting-started guides for {skill}.", 
                            "resource": f"Official {skill} Docs"
                        },
                        {
                            "step": 2, 
                            "action": f"Build a small standalone mini-project to practice core concepts of {skill}.", 
                            "resource": "Self-Directed Project"
                        }
                    ]
                })
        
        # Sort by priority (highest demand score first)
        # This keeps your high-value curated skills at the top!
        recommendations.sort(key=lambda x: x["demand_score"], reverse=True)
        
        return recommendations[:top_n]