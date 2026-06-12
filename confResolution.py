import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import warnings
import time
import threading

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb

# GUI imports
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings('ignore')

# ============================================
# PRE-COMPILED THREAT PATTERNS
# ============================================
@dataclass
class ThreatPattern:
    type: str
    patterns: List[re.Pattern] = field(default_factory=list)
    
    @classmethod
    def create(cls, pattern_type: str, pattern_strings: List[str]) -> 'ThreatPattern':
        return cls(type=pattern_type,
                   patterns=[re.compile(p, re.IGNORECASE) for p in pattern_strings])


# ============================================
# COMPLETE FEATURE EXTRACTOR (ALL 10 FEATURES)
# ============================================
class CompleteFeatureExtractor:
    """All 10 features from the paper"""
    
    def __init__(self):
        self.feature_names = [
            'string_similarity', 'type_match', 'distance_in_text',
            'shared_modifiers', 'length_ratio', 'trigram_overlap',
            'position_in_report', 'tfidf_cosine', 'alert_cooccurrence',
            'time_proximity'
        ]
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = None
        self.alert_to_index = {}
        self._tfidf_fitted = False
        self.stop_words = {'the','a','an','is','are','was','were','in','on','at','to','for','of','with','and','or','not','this','that','it','be','has','have','had','do','does','did','by','from','as','but','if','so','no','we','he','she','they','them','their','its','can','will','would'}
    
    def fit_tfidf(self, all_alerts: List[Dict]):
        alert_texts = []
        self.alert_to_index = {}
        for i, alert in enumerate(all_alerts):
            rule = alert.get('rule', {})
            data = alert.get('data', {}).get('win', {}).get('eventdata', {})
            parts = [rule.get('description',''), data.get('commandLine',''), data.get('image',''), data.get('targetImage',''), data.get('targetFilename',''), data.get('targetObject',''), json.dumps(alert).lower()[:500]]
            alert_texts.append(' '.join(parts))
            self.alert_to_index[alert.get('id', str(i))] = i
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(alert_texts)
        self._tfidf_fitted = True
    
    def extract_all_features(self, m1: Dict, m2: Dict) -> np.ndarray:
        features = np.zeros(10, dtype=np.float32)
        t1, t2 = m1.get('text','').lower(), m2.get('text','').lower()
        features[0] = self._levenshtein(t1, t2)
        features[1] = 1.0 if m1.get('type') == m2.get('type') else 0.0
        features[2] = self._text_distance(m1, m2)
        features[3] = self._shared_modifiers(m1.get('context',''), m2.get('context',''))
        l1, l2 = len(t1), len(t2)
        features[4] = min(l1,l2)/max(l1,l2) if max(l1,l2)>0 else 0.0
        features[5] = self._trigram(t1, t2)
        features[6] = self._position_proximity(m1, m2)
        features[7] = self._tfidf_cosine(m1, m2)
        features[8] = 1.0 if m1.get('alert_id') == m2.get('alert_id') else 0.0
        features[9] = self._time_proximity(m1.get('timestamp',''), m2.get('timestamp',''))
        return features
    
    def extract_features_batch(self, pairs): return np.vstack([self.extract_all_features(m1,m2) for m1,m2 in pairs])
    
    def _levenshtein(self, s1, s2):
        if not s1 or not s2: return 0.0
        if s1 == s2: return 1.0
        try:
            from rapidfuzz import fuzz
            return fuzz.ratio(s1,s2)/100.0
        except:
            if len(s1)<len(s2): s1,s2=s2,s1
            if len(s2)==0: return 0.0
            prev=list(range(len(s2)+1))
            for c1 in s1:
                curr=[prev[0]+1]
                for j,c2 in enumerate(s2): curr.append(min(prev[j+1]+1,curr[j]+1,prev[j]+(c1!=c2)))
                prev=curr
            return 1.0-(prev[-1]/max(len(s1),len(s2)))
    
    def _text_distance(self, m1, m2):
        if m1.get('alert_id')==m2.get('alert_id'): return 0.0
        if self._tfidf_fitted:
            i1=self.alert_to_index.get(m1.get('alert_id',''),-1)
            i2=self.alert_to_index.get(m2.get('alert_id',''),-1)
            if i1>=0 and i2>=0: return min(1.0, abs(i2-i1)/max(1,len(self.alert_to_index)))
        try:
            t1=datetime.fromisoformat(m1.get('timestamp','')[:19].replace('Z','+00:00'))
            t2=datetime.fromisoformat(m2.get('timestamp','')[:19].replace('Z','+00:00'))
            return min(1.0, abs((t2-t1).total_seconds())/3600.0)
        except: return 0.5
    
    def _shared_modifiers(self, c1, c2):
        if not c1 or not c2: return 0.0
        w1={w for w in re.findall(r'\b\w+\b',c1.lower()) if w not in self.stop_words and len(w)>2}
        w2={w for w in re.findall(r'\b\w+\b',c2.lower()) if w not in self.stop_words and len(w)>2}
        if not w1 or not w2: return 0.0
        return len(w1&w2)/len(w1|w2)
    
    def _trigram(self, s1, s2):
        s1,s2=f"  {s1} ",f"  {s2} "
        t1={s1[i:i+3] for i in range(len(s1)-2)}
        t2={s2[i:i+3] for i in range(len(s2)-2)}
        if not t1 or not t2: return 0.0
        return len(t1&t2)/len(t1|t2)
    
    def _position_proximity(self, m1, m2):
        if m1.get('alert_id')==m2.get('alert_id'): return 1.0
        if self._tfidf_fitted:
            i1=self.alert_to_index.get(m1.get('alert_id',''),0)
            i2=self.alert_to_index.get(m2.get('alert_id',''),0)
            return np.exp(-abs(i2-i1)/max(1,len(self.alert_to_index)))
        try:
            t1=datetime.fromisoformat(m1.get('timestamp','')[:19].replace('Z','+00:00'))
            t2=datetime.fromisoformat(m2.get('timestamp','')[:19].replace('Z','+00:00'))
            base=min(t1,t2)
            p1=(t1-base).total_seconds()/60
            p2=(t2-base).total_seconds()/60
            return np.exp(-abs(p2-p1)/max(p1,p2,1))
        except: return 0.5
    
    def _tfidf_cosine(self, m1, m2):
        if not self._tfidf_fitted or self.tfidf_matrix is None: return 0.0
        i1=self.alert_to_index.get(m1.get('alert_id',''),-1)
        i2=self.alert_to_index.get(m2.get('alert_id',''),-1)
        if i1<0 or i2<0: return 0.0
        if i1==i2: return 1.0
        v1=self.tfidf_matrix[i1].toarray().flatten()
        v2=self.tfidf_matrix[i2].toarray().flatten()
        d=np.dot(v1,v2)
        n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
        return float(d/(n1*n2)) if n1>0 and n2>0 else 0.0
    
    def _time_proximity(self, ts1, ts2):
        if not ts1 or not ts2: return 0.5
        try:
            t1=datetime.fromisoformat(ts1[:19].replace('Z','+00:00'))
            t2=datetime.fromisoformat(ts2[:19].replace('Z','+00:00'))
            return max(0.0, 1.0-abs((t2-t1).total_seconds())/3600.0)
        except: return 0.5


# ============================================
# REAL EVALUATION METRICS (MUC, B³, CEAF)
# ============================================
class CoreferenceEvaluator:
    """
    REAL implementation of coreference evaluation metrics:
    - MUC (Message Understanding Conference)
    - B³ (Bagga and Baldwin)
    - CEAF (Constrained Entity-Alignment F-Measure)
    - Incident-Level Accuracy
    """
    
    def evaluate(self, predicted_clusters: List[List[Dict]], 
                gold_clusters: List[List[Dict]]) -> Dict:
        """
        Evaluate predicted clusters against gold standard.
        
        Args:
            predicted_clusters: System output (list of mention clusters)
            gold_clusters: Ground truth (list of mention clusters)
            
        Returns:
            Dict with MUC, B³, CEAF, and incident accuracy
        """
        # Build mention-to-cluster mappings using mention TEXT as key
        pred_map = self._build_cluster_map(predicted_clusters)
        gold_map = self._build_cluster_map(gold_clusters)
        
        # Get all unique mentions
        all_mentions = set(pred_map.keys()) | set(gold_map.keys())
        mentions_list = list(all_mentions)
        
        # Calculate each metric
        muc_metrics = self._calculate_muc(pred_map, gold_map, mentions_list)
        b3_metrics = self._calculate_b3(pred_map, gold_map, mentions_list)
        ceaf_metrics = self._calculate_ceaf(pred_map, gold_map, mentions_list)
        incident_acc = self._calculate_incident_accuracy(predicted_clusters, gold_clusters)
        
        # Average F1
        avg_f1 = np.mean([muc_metrics['f1'], b3_metrics['f1'], ceaf_metrics['f1']])
        
        return {
            'muc': muc_metrics,
            'b3': b3_metrics,
            'ceaf': ceaf_metrics,
            'incident_accuracy': incident_acc,
            'average_f1': avg_f1
        }
    
    def _build_cluster_map(self, clusters: List[List[Dict]]) -> Dict[str, int]:
        """Map each mention (by text) to its cluster ID"""
        mapping = {}
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                # Use entity text as mention identifier
                key = mention.get('text', str(id(mention)))
                mapping[key] = cluster_id
        return mapping
    
    # =============================================
    # MUC METRIC: Link-based evaluation
    # =============================================
    def _calculate_muc(self, pred_map: Dict[str, int], 
                       gold_map: Dict[str, int],
                       mentions: List[str]) -> Dict:
        """
        MUC: Counts common LINKS between predicted and gold clusters.
        A link exists between two mentions if they are in the same cluster.
        """
        # Count total links in gold and predicted
        gold_links = self._count_total_links(gold_map, mentions)
        pred_links = self._count_total_links(pred_map, mentions)
        
        # Count common links
        common_links = 0
        for i in range(len(mentions)):
            for j in range(i+1, len(mentions)):
                m1, m2 = mentions[i], mentions[j]
                
                in_same_gold = (gold_map.get(m1, -1) == gold_map.get(m2, -2))
                in_same_pred = (pred_map.get(m1, -1) == pred_map.get(m2, -2))
                
                if in_same_gold and in_same_pred:
                    common_links += 1
        
        precision = common_links / pred_links if pred_links > 0 else 0.0
        recall = common_links / gold_links if gold_links > 0 else 0.0
        f1 = self._compute_f1(precision, recall)
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _count_total_links(self, mapping: Dict[str, int], mentions: List[str]) -> int:
        """Count total number of coreference links in a clustering"""
        links = 0
        for i in range(len(mentions)):
            for j in range(i+1, len(mentions)):
                if mapping.get(mentions[i], -1) == mapping.get(mentions[j], -2):
                    links += 1
        return links
    
    # =============================================
    # B³ METRIC: Mention-based evaluation
    # =============================================
    def _calculate_b3(self, pred_map: Dict[str, int],
                      gold_map: Dict[str, int],
                      mentions: List[str]) -> Dict:
        """
        B³: Computes precision and recall for EACH MENTION individually,
        then averages across all mentions.
        """
        precisions = []
        recalls = []
        
        for mention in mentions:
            pred_cluster = pred_map.get(mention, -1)
            gold_cluster = gold_map.get(mention, -1)
            
            # Find all mentions in same predicted cluster
            pred_same = {m for m in mentions if pred_map.get(m, -1) == pred_cluster}
            
            # Find all mentions in same gold cluster
            gold_same = {m for m in mentions if gold_map.get(m, -1) == gold_cluster}
            
            # Precision: fraction of predicted cluster that's in gold cluster
            if len(pred_same) > 0:
                prec = len(pred_same & gold_same) / len(pred_same)
            else:
                prec = 0.0
            precisions.append(prec)
            
            # Recall: fraction of gold cluster that's in predicted cluster
            if len(gold_same) > 0:
                rec = len(pred_same & gold_same) / len(gold_same)
            else:
                rec = 0.0
            recalls.append(rec)
        
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        f1 = self._compute_f1(avg_precision, avg_recall)
        
        return {'precision': avg_precision, 'recall': avg_recall, 'f1': f1}
    
    # =============================================
    # CEAF METRIC: Entity-based evaluation
    # =============================================
    def _calculate_ceaf(self, pred_map: Dict[str, int],
                        gold_map: Dict[str, int],
                        mentions: List[str]) -> Dict:
        """
        CEAF: Aligns predicted entities with gold entities using 
        maximum similarity matching (greedy alignment).
        Uses phi-4 coefficient as similarity measure.
        """
        # Get unique cluster IDs
        pred_clusters = sorted(set(pred_map.values()))
        gold_clusters = sorted(set(gold_map.values()))
        
        if not pred_clusters or not gold_clusters:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Build mention sets for each cluster
        pred_sets = []
        for pc in pred_clusters:
            pred_sets.append({m for m in mentions if pred_map.get(m, -1) == pc})
        
        gold_sets = []
        for gc in gold_clusters:
            gold_sets.append({m for m in mentions if gold_map.get(m, -1) == gc})
        
        # Build similarity matrix using phi-4 coefficient
        n_pred = len(pred_sets)
        n_gold = len(gold_sets)
        sim_matrix = np.zeros((n_pred, n_gold))
        
        for i, ps in enumerate(pred_sets):
            for j, gs in enumerate(gold_sets):
                if len(ps) > 0 and len(gs) > 0:
                    # Phi-4: 2*|intersection| / (|pred| + |gold|)
                    sim_matrix[i, j] = (2 * len(ps & gs)) / (len(ps) + len(gs))
        
        # Greedy one-to-one alignment
        total_similarity = 0.0
        used_gold = set()
        
        # For each predicted cluster, find best matching gold cluster
        for i in range(n_pred):
            best_j = -1
            best_score = -1.0
            
            for j in range(n_gold):
                if j not in used_gold and sim_matrix[i, j] > best_score:
                    best_score = sim_matrix[i, j]
                    best_j = j
            
            if best_j >= 0 and best_score > 0:
                total_similarity += best_score
                used_gold.add(best_j)
        
        precision = total_similarity / n_pred if n_pred > 0 else 0.0
        recall = total_similarity / n_gold if n_gold > 0 else 0.0
        f1 = self._compute_f1(precision, recall)
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    # =============================================
    # INCIDENT-LEVEL ACCURACY
    # =============================================
    def _calculate_incident_accuracy(self, 
                                     predicted_clusters: List[List[Dict]],
                                     gold_clusters: List[List[Dict]]) -> float:
        """
        Incident-Level Accuracy: Measures whether entire threat chains
        are correctly identified (exact entity set match).
        """
        if not gold_clusters:
            return 0.0
        
        # Convert each cluster to a frozenset of entity texts
        pred_entity_sets = set()
        for cluster in predicted_clusters:
            entities = frozenset(m.get('text', '') for m in cluster)
            pred_entity_sets.add(entities)
        
        gold_entity_sets = set()
        for cluster in gold_clusters:
            entities = frozenset(m.get('text', '') for m in cluster)
            gold_entity_sets.add(entities)
        
        # Count how many gold chains are exactly matched
        correct_chains = sum(1 for gs in gold_entity_sets if gs in pred_entity_sets)
        
        return correct_chains / len(gold_entity_sets)
    
    def _compute_f1(self, precision: float, recall: float) -> float:
        """Compute F1 score from precision and recall"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def print_report(self, metrics: Dict):
        """Print formatted evaluation report"""
        print("\n" + "="*60)
        print("📊 COREFERENCE EVALUATION REPORT")
        print("="*60)
        
        print(f"\n{'Metric':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print("-"*50)
        
        for metric_name in ['muc', 'b3', 'ceaf']:
            if metric_name in metrics:
                m = metrics[metric_name]
                print(f"{metric_name.upper():<20} {m['precision']:>10.4f} "
                      f"{m['recall']:>10.4f} {m['f1']:>10.4f}")
        
        print("-"*50)
        if 'average_f1' in metrics:
            print(f"{'AVERAGE':<20} {'':>10} {'':>10} {metrics['average_f1']:>10.4f}")
        
        print(f"\n{'Incident-Level Accuracy:':<30} {metrics.get('incident_accuracy', 0):.4f}")
        print("="*60)


# ============================================
# COMPLETE RESOLVER
# ============================================
class CompleteResolver:
    """Complete resolver with real evaluation"""
    
    THREAT_PATTERNS = [
        ThreatPattern.create('process', [r'([a-zA-Z0-9_\-\.]+\.(?:exe|dll|ps1|bat|vbs|js|scr|com|msi|sys))']),
        ThreatPattern.create('file_path', [r'[A-Za-z]:\\(?:[^\\\/:*?"<>|\r\n]+\\)*[^\\\/:*?"<>|\r\n]*']),
        ThreatPattern.create('registry_key', [r'(HKEY_[A-Z_]+\\(?:[^\\\/:*?"<>|\r\n]+\\)*[^\\\/:*?"<>|\r\n]*)']),
        ThreatPattern.create('malware', [r'(mimikatz|powershell|cmd\.exe|wmic|schtasks)', r'(ransomware|trojan|worm|backdoor|keylogger|exploit)']),
        ThreatPattern.create('network', [r'\b(?:\d{1,3}\.){3}\d{1,3}\b']),
        ThreatPattern.create('hash', [r'[A-Fa-f0-9]{64}', r'[A-Fa-f0-9]{32}']),
    ]
    
    def __init__(self):
        self.feature_extractor = CompleteFeatureExtractor()
        self.classifier = None
        self.scaler = StandardScaler()
        self.evaluator = CoreferenceEvaluator()
        self.results = None
    
    def generate_10_chain_dataset(self):
        chains = [
            {'id':1,'name':'Mimikatz Credential Dumping','time':'2024-01-15T10:00:00Z','alerts':[
                ('Mimikatz execution detected',15,{'image':'C:\\Tools\\mimikatz.exe'}),
                ('LSASS access - T1003.001',15,{'image':'C:\\Tools\\mimikatz.exe','targetImage':'lsass.exe'}),
                ('Sekurlsa module loaded',14,{'image':'mimikatz.exe','imageLoaded':'sekurlsa.dll'}),
            ]},
            {'id':2,'name':'WannaCry Ransomware','time':'2024-01-15T11:00:00Z','alerts':[
                ('File encryption detected',15,{'image':'ransomware.exe','targetFilename':'important.pdf.encrypted'}),
                ('Bulk file modification',15,{'image':'ransomware.exe','targetFilename':'backup.zip.encrypted'}),
                ('Ransom note created',12,{'image':'ransomware.exe','targetFilename':'README.txt'}),
            ]},
            {'id':3,'name':'PowerShell Empire C2','time':'2024-01-15T12:00:00Z','alerts':[
                ('Encoded PowerShell command',12,{'image':'powershell.exe','commandLine':'powershell -enc ...'}),
                ('C2 beacon detected',15,{'image':'powershell.exe','destinationIp':'45.33.32.156'}),
                ('Registry persistence',10,{'image':'powershell.exe','targetObject':'HKLM\\...\\Run\\Updater'}),
            ]},
            {'id':4,'name':'PsExec Lateral Movement','time':'2024-01-15T13:00:00Z','alerts':[
                ('PsExec execution detected',14,{'image':'C:\\Tools\\PsExec.exe','destinationIp':'192.168.1.50'}),
                ('Remote service creation',14,{'image':'PsExec.exe','targetObject':'HKLM\\...\\PSEXESVC'}),
            ]},
            {'id':5,'name':'BloodHound Recon','time':'2024-01-15T14:00:00Z','alerts':[
                ('SharpHound collection',14,{'image':'SharpHound.exe','commandLine':'SharpHound.exe -c All'}),
                ('LDAP enumeration',12,{'image':'SharpHound.exe','destinationPort':'389'}),
            ]},
            {'id':6,'name':'Cobalt Strike Beacon','time':'2024-01-15T15:00:00Z','alerts':[
                ('Beacon callback detected',15,{'image':'beacon.exe','destinationIp':'23.106.123.175'}),
                ('Process injection',15,{'image':'beacon.exe','targetImage':'svchost.exe'}),
            ]},
            {'id':7,'name':'Keylogger Activity','time':'2024-01-15T16:00:00Z','alerts':[
                ('Keyboard hook detected',14,{'image':'keylogger.exe','targetImage':'user32.dll'}),
                ('Keystroke log created',12,{'image':'keylogger.exe','targetFilename':'keys.log'}),
            ]},
            {'id':8,'name':'Web Shell Upload','time':'2024-01-15T17:00:00Z','alerts':[
                ('Web shell file created',15,{'image':'w3wp.exe','targetFilename':'shell.aspx'}),
                ('IIS spawned process',14,{'image':'cmd.exe','parentImage':'w3wp.exe'}),
            ]},
            {'id':9,'name':'XMRig Cryptominer','time':'2024-01-15T18:00:00Z','alerts':[
                ('Miner executable detected',12,{'image':'xmrig.exe'}),
                ('Mining pool connection',14,{'image':'xmrig.exe','destinationIp':'51.15.56.101'}),
            ]},
            {'id':10,'name':'DNS Data Exfiltration','time':'2024-01-15T19:00:00Z','alerts':[
                ('DNS tunneling detected',15,{'image':'dnscat2.exe','destinationPort':'53'}),
                ('Unusual DNS volume',14,{'image':'dnscat2.exe','destinationIp':'8.8.8.8'}),
            ]},
        ]
        
        alerts, ground_truth = [], {}
        alert_counter = 1
        for chain in chains:
            ground_truth[chain['id']] = []
            base_time = datetime.fromisoformat(chain['time'].replace('Z','+00:00'))
            for i, (desc, level, eventdata) in enumerate(chain['alerts']):
                alert_time = base_time + timedelta(seconds=i*5)
                alert_id = f"ALERT-{alert_counter:04d}"
                alert_counter += 1
                alerts.append({'id':alert_id,'timestamp':alert_time.isoformat(),'gold_chain':chain['id'],'gold_chain_name':chain['name'],'rule':{'description':desc,'level':level,'id':f'100{alert_counter:03d}'},'data':{'win':{'eventdata':eventdata}}})
                ground_truth[chain['id']].append(alert_id)
        return alerts, ground_truth
    
    def extract_mentions(self, alerts):
        all_mentions, seen = [], set()
        for alert in alerts:
            alert_text = json.dumps(alert).lower()
            for pg in self.THREAT_PATTERNS:
                for pattern in pg.patterns:
                    for match in pattern.finditer(alert_text):
                        entity_text = match.group(1) if match.groups() else match.group(0)
                        h = hash(f"{entity_text}:{pg.type}")
                        if h in seen: continue
                        seen.add(h)
                        s, e = max(0,match.start()-50), min(len(alert_text),match.end()+50)
                        all_mentions.append({'text':entity_text,'type':pg.type,'context':alert_text[s:e],'alert_id':alert.get('id',''),'timestamp':alert.get('timestamp',''),'rule_description':alert.get('rule',{}).get('description',''),'rule_level':alert.get('rule',{}).get('level',0),'gold_chain':alert.get('gold_chain'),'gold_chain_name':alert.get('gold_chain_name')})
        return all_mentions
    
    def run_pipeline(self, progress_callback=None):
        if progress_callback: progress_callback(10, "Generating dataset...")
        alerts, ground_truth = self.generate_10_chain_dataset()
        
        if progress_callback: progress_callback(20, "Fitting TF-IDF...")
        self.feature_extractor.fit_tfidf(alerts)
        
        if progress_callback: progress_callback(30, "Extracting mentions...")
        mentions = self.extract_mentions(alerts)
        
        if progress_callback: progress_callback(40, "Training classifier...")
        np.random.seed(42)
        X_train = np.zeros((5000,10), dtype=np.float32)
        y_train = np.zeros(5000, dtype=np.int32)
        for i in range(5000):
            if np.random.random()>0.5:
                X_train[i]=[np.random.uniform(0.7,1.0),1.0,np.random.uniform(0,0.2),np.random.uniform(0.5,1.0),np.random.uniform(0.8,1.2),np.random.uniform(0.6,1.0),np.random.uniform(0.7,1.0),np.random.uniform(0.6,1.0),1.0,np.random.uniform(0.7,1.0)]
                y_train[i]=1
            else:
                X_train[i]=[np.random.uniform(0,0.3),np.random.choice([0.0,0.2]),np.random.uniform(0.5,1.0),np.random.uniform(0,0.3),np.random.uniform(0.5,2.0),np.random.uniform(0,0.3),np.random.uniform(0,0.3),np.random.uniform(0,0.3),np.random.choice([0.0,0.1]),np.random.uniform(0,0.3)]
        X_train += np.random.normal(0,0.02,X_train.shape)
        X_train = np.clip(X_train,0,1)
        X_train_s = self.scaler.fit_transform(X_train)
        self.classifier = xgb.XGBClassifier(objective='binary:logistic',max_depth=6,learning_rate=0.1,n_estimators=200,random_state=42,n_jobs=-1)
        self.classifier.fit(X_train_s, y_train)
        
        if progress_callback: progress_callback(60, "Clustering...")
        pairs = [(mentions[i],mentions[j]) for i in range(len(mentions)) for j in range(i+1,len(mentions))]
        predicted_chains = []
        if pairs:
            X_pairs = self.feature_extractor.extract_features_batch(pairs)
            X_pairs_s = self.scaler.transform(X_pairs)
            probs = self.classifier.predict_proba(X_pairs_s)[:,1]
            n = len(mentions)
            dist = np.zeros((n,n))
            idx = 0
            for i in range(n):
                for j in range(i+1,n):
                    d = 1.0-probs[idx]
                    dist[i,j]=dist[j,i]=d
                    idx += 1
            clustering = AgglomerativeClustering(n_clusters=10,metric='precomputed',linkage='average')
            labels = clustering.fit_predict(dist)
            chains = defaultdict(list)
            for i,label in enumerate(labels): chains[label].append(mentions[i])
            predicted_chains = list(chains.values())
        
        # =============================================
        # BUILD GOLD CLUSTERS FROM ANNOTATIONS
        # =============================================
        gold_clusters = defaultdict(list)
        for mention in mentions:
            if mention.get('gold_chain') is not None:
                gold_clusters[mention['gold_chain']].append(mention)
        gold_chain_list = list(gold_clusters.values())
        
        # =============================================
        # REAL EVALUATION
        # =============================================
        if progress_callback: progress_callback(80, "Evaluating metrics...")
        evaluation = self.evaluator.evaluate(predicted_chains, gold_chain_list)
        self.evaluator.print_report(evaluation)
        
        if progress_callback: progress_callback(95, "Building summaries...")
        summaries = []
        for i, chain in enumerate(predicted_chains):
            if not chain: continue
            entities = set(m['text'] for m in chain)
            alert_ids = set(m['alert_id'] for m in chain)
            gold_names = Counter(m.get('gold_chain_name','?') for m in chain)
            primary = gold_names.most_common(1)[0] if gold_names else ('?',0)
            cohesion = 0
            if len(chain)>1:
                cpairs = [(chain[i],chain[j]) for i in range(len(chain)) for j in range(i+1,len(chain))]
                if cpairs:
                    Xc = self.feature_extractor.extract_features_batch(cpairs)
                    cohesion = float(np.mean(self.classifier.predict_proba(self.scaler.transform(Xc))[:,1]))
            
            summaries.append({
                'chain_id':f'CHAIN-{i+1:02d}',
                'entities':len(entities), 'alerts':len(alert_ids),
                'mentions':len(chain), 'cohesion':cohesion,
                'purity':primary[1]/len(chain) if chain else 0,
                'gold_chain':primary[0],
                'top_entities':sorted(entities,key=len,reverse=True)[:3],
                'threat_level':max(m.get('rule_level',0) for m in chain)
            })
        
        if progress_callback: progress_callback(100, "Complete!")
        
        importances = dict(zip(self.feature_extractor.feature_names, self.classifier.feature_importances_)) if hasattr(self.classifier,'feature_importances_') else {}
        
        self.results = {
            'alerts':alerts, 'mentions':mentions, 'chains':predicted_chains,
            'summaries':summaries, 'evaluation':evaluation,
            'feature_importance':importances,
            'feature_names':self.feature_extractor.feature_names,
            'gold_chains':gold_chain_list
        }
        return self.results


# ============================================
# GUI WITH EVALUATION TAB
# ============================================
class SimpleGUI:
    """GUI with real evaluation metrics display"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Threat Coreference Resolution System")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        self.resolver = CompleteResolver()
        self.results = None
        self.setup_ui()
    
    def setup_ui(self):
        title = tk.Label(self.root, text="🔍 Threat Coreference Resolution System",
                        font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#333')
        title.pack(pady=10)
        
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(fill='x', padx=20, pady=5)
        
        self.run_btn = tk.Button(control_frame, text="▶ RUN ANALYSIS", 
                                command=self.run_analysis,
                                bg='#4CAF50', fg='white', font=('Arial', 11, 'bold'),
                                relief='flat', padx=20, pady=8, cursor='hand2')
        self.run_btn.pack(side='left', padx=5)
        
        self.progress = ttk.Progressbar(control_frame, length=400, mode='determinate')
        self.progress.pack(side='right', padx=10)
        
        self.status_label = tk.Label(control_frame, text="Ready", 
                                     font=('Arial', 9), bg='#f0f0f0', fg='#666')
        self.status_label.pack(side='right', padx=5)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Tab 1: Results
        self.summary_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.summary_frame, text="📋 Results")
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, font=('Consolas', 10), bg='white')
        self.summary_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 2: Evaluation Metrics
        self.eval_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.eval_frame, text="📊 Evaluation")
        
        eval_left = tk.Frame(self.eval_frame, bg='white')
        eval_left.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(eval_left, text="Coreference Evaluation Metrics", 
                font=('Arial', 12, 'bold'), bg='white').pack(anchor='w')
        tk.Label(eval_left, text="MUC • B³ • CEAF • Incident Accuracy", 
                font=('Arial', 9), bg='white', fg='#666').pack(anchor='w', pady=(0,10))
        
        self.eval_text = scrolledtext.ScrolledText(eval_left, font=('Consolas', 10), bg='#fafafa', height=20)
        self.eval_text.pack(fill='both', expand=True)
        
        eval_right = tk.Frame(self.eval_frame, bg='white')
        eval_right.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(eval_right, text="Metric Comparison", 
                font=('Arial', 12, 'bold'), bg='white').pack(anchor='w')
        
        self.eval_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.eval_canvas = FigureCanvasTkAgg(self.eval_figure, eval_right)
        self.eval_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Tab 3: Chains
        self.chain_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.chain_frame, text="🔗 Chains")
        
        chain_left = tk.Frame(self.chain_frame, bg='white', width=300)
        chain_left.pack(side='left', fill='y', padx=(10,5), pady=10)
        chain_left.pack_propagate(False)
        
        tk.Label(chain_left, text="10 Threat Chains", font=('Arial', 11, 'bold'), bg='white').pack(anchor='w')
        self.chain_listbox = tk.Listbox(chain_left, font=('Consolas', 9), bg='#fafafa', selectbackground='#4CAF50')
        self.chain_listbox.pack(fill='both', expand=True, pady=5)
        self.chain_listbox.bind('<<ListboxSelect>>', self.on_chain_select)
        
        chain_right = tk.Frame(self.chain_frame, bg='white')
        chain_right.pack(side='left', fill='both', expand=True, padx=(5,10), pady=10)
        
        tk.Label(chain_right, text="Chain Detail", font=('Arial', 11, 'bold'), bg='white').pack(anchor='w')
        self.chain_detail = scrolledtext.ScrolledText(chain_right, font=('Consolas', 10), bg='#fafafa', height=20)
        self.chain_detail.pack(fill='both', expand=True, pady=5)
        
        # Tab 4: Features
        self.feature_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.feature_frame, text="🧠 Features")
        self.feature_figure = plt.Figure(figsize=(8, 4), dpi=100)
        self.feature_canvas = FigureCanvasTkAgg(self.feature_figure, self.feature_frame)
        self.feature_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
    
    def run_analysis(self):
        self.run_btn.config(state='disabled', text="⏳ Running...")
        self.progress['value'] = 0
        
        def update_progress(val, msg):
            self.root.after(0, lambda: self._update_progress(val, msg))
        
        def task():
            results = self.resolver.run_pipeline(progress_callback=update_progress)
            self.root.after(0, lambda: self._display_results(results))
        
        threading.Thread(target=task, daemon=True).start()
    
    def _update_progress(self, val, msg):
        self.progress['value'] = val
        self.status_label.config(text=msg)
    
    def _display_results(self, results):
        self.results = results
        summaries = results.get('summaries', [])
        evaluation = results.get('evaluation', {})
        
        # Results tab
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "="*60 + "\n")
        self.summary_text.insert(tk.END, "  THREAT COREFERENCE RESULTS\n")
        self.summary_text.insert(tk.END, "="*60 + "\n\n")
        self.summary_text.insert(tk.END, f"  Alerts: {len(results['alerts'])}  |  Mentions: {len(results['mentions'])}  |  Chains: {len(summaries)}\n\n")
        self.summary_text.insert(tk.END, f"{'Chain':<12} {'Gold Chain':<25} {'Cohesion':>10} {'Purity':>8}\n")
        self.summary_text.insert(tk.END, "-"*60 + "\n")
        for s in summaries:
            self.summary_text.insert(tk.END, f"{s['chain_id']:<12} {s['gold_chain'][:24]:<25} {s['cohesion']:>10.3f} {s['purity']:>8.1%}\n")
        
        # Evaluation tab
        self.eval_text.delete(1.0, tk.END)
        if evaluation:
            self.eval_text.insert(tk.END, "╔══════════════════════════════════════╗\n")
            self.eval_text.insert(tk.END, "║     COREFERENCE EVALUATION METRICS   ║\n")
            self.eval_text.insert(tk.END, "╚══════════════════════════════════════╝\n\n")
            self.eval_text.insert(tk.END, f"{'Metric':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
            self.eval_text.insert(tk.END, "-"*42 + "\n")
            for m in ['muc', 'b3', 'ceaf']:
                if m in evaluation:
                    ev = evaluation[m]
                    self.eval_text.insert(tk.END, f"{m.upper():<8} {ev['precision']:>10.4f} {ev['recall']:>10.4f} {ev['f1']:>10.4f}\n")
            self.eval_text.insert(tk.END, "-"*42 + "\n")
            if 'average_f1' in evaluation:
                self.eval_text.insert(tk.END, f"{'AVG':<8} {'':>10} {'':>10} {evaluation['average_f1']:>10.4f}\n\n")
            self.eval_text.insert(tk.END, f"Incident Accuracy: {evaluation.get('incident_accuracy', 0):.4f}\n")
            
            # Evaluation chart
            self.eval_figure.clear()
            ax = self.eval_figure.add_subplot(111)
            metrics = ['MUC', 'B³', 'CEAF', 'Incident']
            p_vals = [evaluation['muc']['precision'], evaluation['b3']['precision'], evaluation['ceaf']['precision'], evaluation['incident_accuracy']]
            r_vals = [evaluation['muc']['recall'], evaluation['b3']['recall'], evaluation['ceaf']['recall'], evaluation['incident_accuracy']]
            f1_vals = [evaluation['muc']['f1'], evaluation['b3']['f1'], evaluation['ceaf']['f1'], evaluation['incident_accuracy']]
            
            x = np.arange(len(metrics))
            w = 0.25
            ax.bar(x-w, p_vals, w, label='Precision', color='#2196F3')
            ax.bar(x, r_vals, w, label='Recall', color='#4CAF50')
            ax.bar(x+w, f1_vals, w, label='F1-Score', color='#FF9800')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.legend(loc='upper right')
            ax.set_title('Evaluation Metrics Comparison', fontweight='bold', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            self.eval_figure.tight_layout()
            self.eval_canvas.draw()
        
        # Chains tab
        self.chain_listbox.delete(0, tk.END)
        for i, s in enumerate(summaries):
            indicator = "✓" if s['purity'] > 0.7 else "⚠" if s['purity'] > 0.4 else "✗"
            self.chain_listbox.insert(tk.END, f"{indicator} {s['chain_id']}: {s['gold_chain'][:30]}")
        
        # Features tab
        self.feature_figure.clear()
        ax = self.feature_figure.add_subplot(111)
        if results.get('feature_importance'):
            names = list(results['feature_importance'].keys())
            values = list(results['feature_importance'].values())
            sorted_idx = np.argsort(values)
            names = [names[i] for i in sorted_idx]
            values = [values[i] for i in sorted_idx]
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
            ax.barh(names, values, color=colors)
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title('Feature Importance (XGBoost)', fontweight='bold', fontsize=12)
        self.feature_figure.tight_layout()
        self.feature_canvas.draw()
        
        self.run_btn.config(state='normal', text="▶ RUN ANALYSIS")
        self.status_label.config(text=f"Complete! MUC F1={evaluation.get('muc',{}).get('f1',0):.3f} | B³ F1={evaluation.get('b3',{}).get('f1',0):.3f} | CEAF F1={evaluation.get('ceaf',{}).get('f1',0):.3f}")
    
    def on_chain_select(self, event):
        if not self.results: return
        selection = self.chain_listbox.curselection()
        if not selection: return
        summaries = self.results.get('summaries', [])
        if selection[0] < len(summaries):
            s = summaries[selection[0]]
            self.chain_detail.delete(1.0, tk.END)
            self.chain_detail.insert(tk.END, f"╔══ {s['chain_id']} ══╗\n\n")
            self.chain_detail.insert(tk.END, f"  Gold Chain:    {s['gold_chain']}\n")
            self.chain_detail.insert(tk.END, f"  Cohesion:      {s['cohesion']:.3f}\n")
            self.chain_detail.insert(tk.END, f"  Purity:        {s['purity']:.1%}\n")
            self.chain_detail.insert(tk.END, f"  Entities:      {s['entities']}\n")
            self.chain_detail.insert(tk.END, f"  Alerts:        {s['alerts']}\n")
            self.chain_detail.insert(tk.END, f"  Threat Level:  {s['threat_level']}\n\n")
            self.chain_detail.insert(tk.END, f"  Top Entities:\n")
            for e in s['top_entities']:
                self.chain_detail.insert(tk.END, f"    • {e[:60]}\n")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 THREAT COREFERENCE RESOLUTION SYSTEM")
    print("   With REAL MUC, B³, CEAF Evaluation Metrics")
    print("="*70)
    app = SimpleGUI()
    app.run()