"""Check that every entry of a .bib file exists: DOI entries against their Crossref record (title,
first author, year, venue); entries without a DOI against arXiv or Semantic Scholar by title.
Usage: python check_refs.py references.bib OUT.json"""
import difflib, json, re, subprocess, sys, time, unicodedata, urllib.parse

BIB, OUT = sys.argv[1], sys.argv[2]
text = open(BIB, encoding='utf-8').read()

def entries(t):
    i = 0
    while True:
        m = re.compile(r'@(\w+)\s*\{\s*([^,\s]+)\s*,').search(t, i)
        if not m:
            return
        j, depth = m.end(), 1
        while depth:
            depth += {'{': 1, '}': -1}.get(t[j], 0); j += 1
        body = t[m.end():j - 1]
        fields, k = {}, 0
        for fm in re.finditer(r'(\w+)\s*=\s*\{', body):
            if fm.start() < k:
                continue
            a, d = fm.end(), 1
            while d:
                d += {'{': 1, '}': -1}.get(body[a], 0); a += 1
            fields[fm.group(1).lower()] = body[fm.end():a - 1]
            k = a
        yield m.group(1).lower(), m.group(2), fields
        i = j

def clean(s):
    s = re.sub(r'\\[a-zA-Z]+\s*', '', s or '')
    s = re.sub(r'[{}\\$^"\'`~]', '', s)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()

def sim(a, b):
    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()

def get(url, accept=None):
    cmd = ['curl', '-sS', '-m', '40', '-A', 'reference-check/1.0']
    if accept:
        cmd += ['-H', 'Accept: ' + accept]
    for attempt in range(3):
        r = subprocess.run(cmd + [url], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
        time.sleep(2 + 3 * attempt)
    return ''

def first_author(fields):
    a = fields.get('author', '').split(' and ')[0]
    return clean(a.split(',')[0] if ',' in a else a.split()[-1] if a.split() else '')

out = []
for typ, key, f in entries(text):
    rec = dict(key=key, title=f.get('title', ''), year=f.get('year', ''), doi=f.get('doi', ''))
    if rec['doi']:
        raw = get('https://api.crossref.org/works/' + urllib.parse.quote(rec['doi']))
        try:
            m = json.loads(raw)['message']
            t = (m.get('title') or [''])[0]
            yrs = [p['date-parts'][0][0] for p in [m.get('published-print'), m.get('published-online'), m.get('issued')] if p and p.get('date-parts') and p['date-parts'][0][0]]
            fa = clean(((m.get('author') or [{}])[0]).get('family', ''))
            rec.update(source='crossref', found_title=t, title_sim=round(sim(rec['title'], t), 3), found_years=sorted(set(yrs)),
                       found_first_author=fa, first_author_ok=(fa == first_author(f) or fa in first_author(f) or first_author(f) in fa),
                       venue=(m.get('container-title') or [''])[-1])
        except Exception as e:
            rec.update(source='crossref', error='no record (%s)' % type(e).__name__)
    else:
        # no DOI: the arXiv record by title (HTTPS; entries list <published> before <title>), with the
        # version comment and journal reference, which name the venue when the authors gave it
        raw = get('https://export.arxiv.org/api/query?max_results=5&search_query=ti:'
                  + urllib.parse.quote('"%s"' % clean(rec['title'])))
        best = None
        for e in raw.split('<entry>')[1:]:
            g = lambda pat: (re.search(pat, e, re.S).group(1).strip() if re.search(pat, e, re.S) else '')
            t = ' '.join(g(r'<title>(.*?)</title>').split())
            cand = (sim(rec['title'], t), t, g(r'<published>(\d{4})'), g(r'<name>(.*?)</name>'), g(r'<id>(.*?)</id>'),
                    ' '.join(g(r'<arxiv:comment[^>]*>(.*?)</arxiv:comment>').split()),
                    ' '.join(g(r'<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>').split()))
            if best is None or cand[0] > best[0]:
                best = cand
        if best and best[0] > 0.9:
            rec.update(source='arxiv', found_title=best[1], title_sim=round(best[0], 3), found_years=[int(best[2])],
                       found_first_author=clean(best[3].split()[-1]), arxiv_id=best[4].split('/abs/')[-1],
                       first_author_ok=clean(best[3].split()[-1]) in first_author(f) or first_author(f) in clean(best[3]),
                       arxiv_comment=best[5], journal_ref=best[6], venue=f.get('booktitle', f.get('howpublished', '')))
        else:
            rec.update(source='none', error='not found on arXiv')
        time.sleep(3)                                   # arXiv asks for a pause between queries
    out.append(rec)
    print('%-26s %-15s sim %-5s years %-12s %s' % (key, rec.get('source'), rec.get('title_sim'), rec.get('found_years'),
                                                   rec.get('error', '')), flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
