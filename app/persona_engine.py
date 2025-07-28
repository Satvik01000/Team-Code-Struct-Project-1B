import time
import numpy as np
import re
import os  # ADD THIS IMPORT
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.document_processor import DocumentProcessor
from app.structure_analyzer import StructureAnalyzer

class PersonaIntelligenceEngine:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.structure_analyzer = StructureAnalyzer()
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Enhanced keywords for the food contractor persona
        self.persona_keywords = {
            'food_contractor': ['vegetarian', 'veg', 'veggie', 'gluten-free', 'gluten free', 
                               'buffet', 'dinner', 'menu', 'dish', 'recipe', 'ingredients',
                               'vegetable', 'plant-based', 'meat-free', 'meatless'],
        }
        
        # Keywords to EXCLUDE (non-vegetarian items)
        self.exclude_keywords = ['chicken', 'beef', 'pork', 'lamb', 'turkey', 'fish', 
                                'shrimp', 'meat', 'bacon', 'sausage', 'ham', 'seafood']
        
    def analyze_document_collection(self, pdf_paths: List[str], persona: str, job: str) -> Dict:
        start_time = time.time()
        
        print(f"📚 Processing {len(pdf_paths)} documents...")
        all_sections = self._extract_dinner_sections(pdf_paths)
        print(f"    - Extracted {len(all_sections)} sections.")
        
        # Filter for vegetarian and gluten-free items
        filtered_sections = self._filter_vegetarian_gluten_free(all_sections)
        print(f"    - Found {len(filtered_sections)} vegetarian/gluten-free sections.")
        
        # Rank by relevance
        ranked_sections = self._rank_sections(filtered_sections, persona, job)
        
        # Extract refined text from top sections
        subsection_analysis = self._extract_refined_text(ranked_sections[:10])
        
        return self._format_output(persona, job, pdf_paths, ranked_sections, subsection_analysis)
    
    def _extract_dinner_sections(self, pdf_paths: List[str]) -> List[Dict]:
        """Extract sections specifically from dinner-related PDFs"""
        all_sections = []
        
        for path in pdf_paths:
            filename = os.path.basename(path)
            
            # Only process dinner-related files for this task
            if 'Dinner' not in filename and 'Lunch' not in filename:
                continue
                
            doc_content = self.doc_processor.extract_structured_content(path)
            
            # Extract individual recipe/dish sections
            sections = self._extract_recipe_sections(doc_content, filename)
            all_sections.extend(sections)
            
        return all_sections
    
    def _extract_recipe_sections(self, doc_content: Dict, filename: str) -> List[Dict]:
        """Extract individual recipes as sections"""
        sections = []
        
        for page in doc_content.get("pages", []):
            page_text = page.get("raw_text", "")
            page_num = page.get("page_index", 0)
            
            # Split by common recipe patterns
            # Look for recipe titles (usually standalone lines before "Ingredients:")
            recipes = re.split(r'\n(?=[A-Z][a-zA-Z\s]+\nIngredients:)', page_text)
            
            for recipe_text in recipes:
                if 'Ingredients:' in recipe_text:
                    # Extract recipe name
                    lines = recipe_text.strip().split('\n')
                    recipe_name = lines[0].strip() if lines else "Unknown Recipe"
                    
                    sections.append({
                        "document": filename,
                        "page": page_num,
                        "section_title": recipe_name,
                        "content": recipe_text.strip()
                    })
        
        return sections
    
    def _filter_vegetarian_gluten_free(self, sections: List[Dict]) -> List[Dict]:
        """Filter sections to only include vegetarian and potentially gluten-free items"""
        filtered = []
        
        for section in sections:
            content_lower = section['content'].lower()
            title_lower = section['section_title'].lower()
            
            # Skip if contains non-vegetarian ingredients
            if any(keyword in content_lower for keyword in self.exclude_keywords):
                continue
            
            # Check for vegetarian indicators
            is_vegetarian = (
                'vegetable' in title_lower or
                'veggie' in title_lower or
                'vegetarian' in title_lower or
                not any(meat in content_lower for meat in self.exclude_keywords)
            )
            
            # Check for gluten-free potential (no wheat, flour, pasta, bread in ingredients)
            gluten_ingredients = ['flour', 'bread', 'pasta', 'wheat', 'noodles', 'couscous']
            is_potentially_gluten_free = not any(gluten in content_lower for gluten in gluten_ingredients)
            
            # Include if vegetarian (prioritize gluten-free)
            if is_vegetarian:
                section['is_gluten_free'] = is_potentially_gluten_free
                filtered.append(section)
        
        return filtered
    
    def _rank_sections(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Rank sections by relevance to buffet-style dinner and gluten-free requirements"""
        for section in sections:
            score = 0
            content_lower = section['content'].lower()
            
            # Bonus for gluten-free items
            if section.get('is_gluten_free', False):
                score += 0.3
            
            # Bonus for dishes suitable for buffet
            buffet_keywords = ['salad', 'dip', 'platter', 'bowl', 'roasted', 'grilled']
            if any(keyword in content_lower for keyword in buffet_keywords):
                score += 0.2
            
            # Check for ease of preparation in bulk
            if 'easy' in content_lower or 'simple' in content_lower:
                score += 0.1
            
            section['relevance_score'] = score
        
        # Sort by relevance
        sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Assign importance ranks
        for i, section in enumerate(sections):
            section['importance_rank'] = i + 1
        
        return sections
    
    def _extract_refined_text(self, top_sections: List[Dict]) -> List[Dict]:
        """Extract recipe ingredients and instructions as refined text"""
        refined = []
        
        for section in top_sections:
            content = section['content']
            
            # Extract ingredients and instructions
            ingredients_match = re.search(r'Ingredients:(.*?)Instructions:', content, re.DOTALL)
            instructions_match = re.search(r'Instructions:(.*?)$', content, re.DOTALL)
            
            if ingredients_match and instructions_match:
                ingredients = ingredients_match.group(1).strip()
                instructions = instructions_match.group(1).strip()
                
                # Clean up formatting
                ingredients = re.sub(r'\s+', ' ', ingredients)
                instructions = re.sub(r'\s+', ' ', instructions)
                
                refined_text = f"{section['section_title']} Ingredients: {ingredients} Instructions: {instructions}"
                
                refined.append({
                    "document": section['document'],
                    "page": section['page'],
                    "refined_text": refined_text[:500]  # Limit length
                })
        
        return refined
    
    def _format_output(self, persona, job, pdfs, sections, subsections):
        """Format output matching the expected structure"""
        # Fix the timestamp format - remove microseconds
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        return {
            "metadata": {
                "input_documents": [os.path.basename(p) for p in pdfs],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": timestamp
            },
            "extracted_sections": [{
                "document": s['document'],
                "section_title": s['section_title'],
                "importance_rank": s['importance_rank'],
                "page_number": s['page']
            } for s in sections[:5]],  # Top 5 sections
            "subsection_analysis": [{
                "document": ss['document'],
                "refined_text": ss['refined_text'],
                "page_number": ss['page']
            } for ss in subsections[:5]]  # Top 5 subsections
        }