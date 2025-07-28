import os
import time
import re
from typing import List, Dict, Any
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

from app.document_processor import DocumentProcessor
from app.structure_analyzer import StructureAnalyzer

NON_VEGETARIAN_TERMS = {'chicken', 'beef', 'pork', 'lamb', 'meat', 'fish', 'tuna', 'salmon', 'bacon', 'sausage'}
GLUTEN_TERMS = {'flour', 'wheat', 'bread', 'pasta', 'barley'}

class PersonaIntelligenceEngine:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.structure_analyzer = StructureAnalyzer()
        model_path = './models/all-MiniLM-L6-v2'
        self.model = SentenceTransformer(model_path)

    def analyze_document_collection(self, pdf_paths: List[str], persona: str, job: str) -> Dict:
        start_time = time.time()
        query_text = f"{persona} {job}"
        
        all_sections = self._get_intelligent_sections(pdf_paths)
        if not all_sections:
            return self._format_final_output(persona, job, pdf_paths, [], [])

        candidate_sections = self._get_semantic_candidates(query_text, all_sections)
        final_ranked_sections = self._filter_and_rank_with_rules(candidate_sections, query_text)
        subsection_analysis = self._extract_subsections(final_ranked_sections[:5])

        return self._format_final_output(persona, job, pdf_paths, final_ranked_sections, subsection_analysis)

    def _get_semantic_candidates(self, query_text: str, all_sections: List[Dict]) -> List[Dict]:
        section_contents = [f"{s['section_title']} {s['content']}" for s in all_sections]
        section_embeddings = self.model.encode(section_contents, convert_to_tensor=True, show_progress_bar=False)
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        
        similarities = util.cos_sim(query_embedding, section_embeddings).flatten()
        
        for i, section in enumerate(all_sections):
            section['relevance_score'] = similarities[i].item()
            
        all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        return all_sections[:100]

    def _filter_and_rank_with_rules(self, candidates: List[Dict], query: str) -> List[Dict]:
        query_lower = query.lower()
        
        filtered_sections = []
        for section in candidates:
            content_lower = (section['section_title'] + ' ' + section['content']).lower()
            
            if 'vegetarian' in query_lower and any(term in content_lower for term in NON_VEGETARIAN_TERMS):
                continue
            if 'gluten-free' in query_lower and any(term in content_lower for term in GLUTEN_TERMS):
                continue
            
            filtered_sections.append(section)
            
        for section in filtered_sections:
            score = section['relevance_score']
            title_lower = section['section_title'].lower()
            
            if 'form' in query_lower and 'form' in title_lower: score *= 1.5
            if 'dinner' in query_lower and 'dinner' in section['document'].lower(): score *= 1.2
            if 'main' in query_lower and 'mains' in section['document'].lower(): score *= 1.2
            
            section['relevance_score'] = score
            
        filtered_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        for i, section in enumerate(filtered_sections):
            section['importance_rank'] = i + 1
            
        return filtered_sections

    def _extract_subsections(self, top_sections: List[Dict]) -> List[Dict]:
        subsections = []
        for section in top_sections:
            sentences = re.split(r'(?<=[.!?])\s+', section['content'])
            snippet = " ".join(sentences[:2])
            
            subsections.append({
                "document": section['document'],
                "page_number": section['page'],
                "refined_text": re.sub(r'\s+', ' ', snippet).strip()[:500]
            })
        return subsections
        
    def _get_intelligent_sections(self, pdf_paths: List[str]) -> List[Dict]:
        all_sections = []
        for path in pdf_paths:
            doc_content = self.doc_processor.extract_structured_content(path)
            _, headings = self.structure_analyzer.analyze(doc_content)
            spans_by_page = defaultdict(list)
            for page in doc_content.get("pages", []): spans_by_page[page['page_index']] = page.get('content_spans', [])
            headings.sort(key=lambda h: (h['page'], h['text']))
            for i, heading in enumerate(headings):
                start_page = heading['page']
                heading_y_pos = 0
                for span in spans_by_page[start_page]:
                    if span['text'].strip() == heading['text']: heading_y_pos = span['style_info']['y1']; break
                end_page = doc_content.get('total_pages', start_page)
                end_y_pos = 9999
                if i + 1 < len(headings):
                    next_heading = headings[i+1]
                    end_page = next_heading['page']
                    for span in spans_by_page[end_page]:
                         if span['text'].strip() == next_heading['text']: end_y_pos = span['style_info']['y0']; break
                section_content_spans = []
                for page_num in range(start_page, end_page + 1):
                    for span in spans_by_page.get(page_num, []):
                        y0 = span['style_info']['y0']
                        if (page_num == start_page and y0 > heading_y_pos) or \
                           (page_num > start_page and page_num < end_page) or \
                           (page_num == end_page and y0 < end_y_pos):
                            section_content_spans.append(span['text'])
                all_sections.append({
                    "document": os.path.basename(path), "page": start_page,
                    "section_title": heading["text"], "content": " ".join(section_content_spans)
                })
        return all_sections

    def _format_final_output(self, persona, job, pdfs, sections, subsections):
        return {
            "metadata": {
                "input_documents": [os.path.basename(p) for p in pdfs],
                "persona": persona, "job_to_be_done": job,
                "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            },
            "extracted_sections": [{
                "document": s['document'], "page_number": s['page'],
                "section_title": s['section_title'], "importance_rank": s['importance_rank']
            } for s in sections],
            "subsection_analysis": subsections
        }