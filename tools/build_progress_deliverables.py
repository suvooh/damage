"""Build evidence-backed exterior model progress and production-planning files."""
from pathlib import Path
import json
import math
import zipfile
from datetime import datetime

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import xlsxwriter
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deliverables'
OUT.mkdir(exist_ok=True)
NAVY='10263D'; INK='172F45'; TEAL='007F82'; GREEN='217950'; AMBER='A9610B'; RED='B64848'; MUTED='5C7080'; LIGHT='F1F5F8'; WHITE='FFFFFF'
DATE='06 November 2026'

def load_rows(path):
    return [json.loads(x) for x in (ROOT/path).read_text(encoding='utf-8').splitlines() if x.strip()]

rows={r['id']:r for r in load_rows('reports/real_pipeline_2026-10-27_v3/v3_all_sets.jsonl')}
for f in ['reports/real_pipeline_2026-10-27_v3/v3_resolution_normalised_changed_images.jsonl','reports/real_pipeline_2026-10-28/v3_glare_grid_flagged_images.jsonl']:
    for r in load_rows(f):
        r['case_score']=r.get('glare_grid',{}).get('0.985|0.025|0.85',r['case_score'])
        rows[r['id']]=r
verified=[]
for label,sel in [('Test damage',lambda r:r['set']=='test' and r['label']=='damage'),('Test clean',lambda r:r['set']=='test' and r['label']=='clean'),('Fresh clean',lambda r:r['set']=='test2' and r['label']=='clean')]:
    group=[r for r in rows.values() if sel(r)]
    verified.append((label,sum(r['case_score']>=.381 for r in group),len(group)))
assert verified==[('Test damage',78,81),('Test clean',0,122),('Fresh clean',11,456)],verified

sources=[
('E01','Selected checkpoint / development status','runs/SELECTED_DEVELOPMENT_MODEL.json','v3 selected; hashes pinned; metrics here predate resolution/glare updates.'),
('E02','Latest readiness record','PRODUCTION_READINESS.md','Latest served configuration dated 2026-10-28; headline metrics and limitations.'),
('E03','Model card / historical type and mask diagnostics','MODEL_CARD.md','32% scratch / 36% paint precision are historical v2 pixel-level diagnostics; current v3 type performance unverified.'),
('E04','Current API and routing','src/vehicle_damage/api.py','Quality, glare review, part assignment, views, panels and report outputs.'),
('E05','Current inference','src/vehicle_damage/inference.py','Working resolution, tiling, TTA, quality gate and glare routing.'),
('E06','Current view rules','src/vehicle_damage/views.py','Left/right/front/rear heuristics implemented; unknown allowed; no measured accuracy.'),
('E07','Severity rules','src/vehicle_damage/severity.py','Dent depth assumed d2; paint as scratch; breakage/deformation use d3 row.'),
('E08','Base v3 scores','reports/real_pipeline_2026-10-27_v3/v3_all_sets.jsonl','Base case scores; merge by image ID with E09 then E10.'),
('E09','Resolution-normalisation score updates','reports/real_pipeline_2026-10-27_v3/v3_resolution_normalised_changed_images.jsonl','Updated cases only; latest record per image ID.'),
('E10','Glare experiment / score updates','reports/real_pipeline_2026-10-28/v3_glare_grid_flagged_images.jsonl','Use glare_grid[0.985|0.025|0.85]. Grid tuned on evaluation photos; optimistic.'),
('E11','Scoring protocol','tools/real_pipeline_scores.py','Case-score region survival, not final API decision; 48 threshold grid.'),
('E12','Rejected v5 scores','reports/real_pipeline_2026-10-28/v5_rejected.jsonl','v5 rejected; 74/81 damage and 33/456 fresh clean flagged per readiness record.'),
('E13','Pipeline example','pipeline_steps/white_scorpio_led/step_12_final_report/result.json','Known LED false region; final decision recapture_required due to low resolution.'),
('E14','Dataset provenance manifests','data_provenance/','Inventory for rights review; no legal/commercial clearance inferred.'),
]

prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
previews=[]
fontroot=Path('C:/Windows/Fonts')
def font(size,bold=False):
    for name in (['aptos-bold.ttf','arialbd.ttf'] if bold else ['aptos.ttf','arial.ttf']):
        if (fontroot/name).exists(): return ImageFont.truetype(str(fontroot/name),int(size*1.5))
    return ImageFont.load_default()
def rect(sl,x,y,w,h,color):
    sh=sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h)); sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor.from_string(color); sh.line.fill.background()
    draw.rectangle((int(x*144),int(y*144),int((x+w)*144),int((y+h)*144)),fill='#'+color)
def text(sl,x,y,w,h,content,size=18,bold=False,color=INK):
    box=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=box.text_frame; tf.word_wrap=True
    tf.margin_left=tf.margin_right=0; tf.margin_top=tf.margin_bottom=0
    for i,line in enumerate(content.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.text=line; p.font.name='Aptos'; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=RGBColor.from_string(color); p.space_after=Pt(8)
    yy=y*144; f=font(size,bold); width=w*144
    for para in content.split('\n'):
        line=''
        for word in para.split():
            candidate=(line+' '+word).strip()
            if draw.textlength(candidate,font=f)>width and line:
                draw.text((x*144,yy),line,font=f,fill='#'+color); yy+=size*2.05; line=word
            else: line=candidate
        draw.text((x*144,yy),line,font=f,fill='#'+color); yy+=size*2.05+12
    return box
def slide(title,subtitle,source):
    global draw,canvas
    s=prs.slides.add_slide(prs.slide_layouts[6]); canvas=Image.new('RGB',(1920,1080),'white'); draw=ImageDraw.Draw(canvas)
    rect(s,0,0,13.333,.12,TEAL); text(s,.55,.33,12.2,.55,title,29,True); text(s,.55,1.02,12.2,.48,subtitle,13,False,MUTED)
    rect(s,.55,6.92,12.2,.014,'D8E1E8'); text(s,.55,7.04,11.5,.3,f'{DATE}  |  Evidence: {source}',9,False,MUTED); text(s,12.2,7.02,.6,.3,f'{len(prs.slides):02}',11,True,TEAL)
    return s
def finish(s,note):
    s.notes_slide.notes_text_frame.text=note
    previews.append(canvas.copy())
def card(s,x,y,w,h,title,body,accent=TEAL,body_size=17):
    rect(s,x,y,w,h,LIGHT); rect(s,x,y,.045,h,accent); text(s,x+.18,y+.16,w-.36,.55,title,19,True,accent); text(s,x+.18,y+.85,w-.36,h-.96,body,body_size)
def picture(s,path,x,y,w,h):
    im=Image.open(ROOT/path).convert('RGB'); im.thumbnail((int(w*144),int(h*144)))
    iw,ih=im.size; xx=x+(w-iw/144)/2; yy=y+(h-ih/144)/2
    s.shapes.add_picture(str(ROOT/path),Inches(xx), Inches(yy),width=Inches(iw/144),height=Inches(ih/144)); canvas.paste(im,(int(xx*144),int(yy*144)))

s=slide('Exterior damage model | progress review','Current implementation, measured results and the path to production','E01, E02, E08–E11')
rect(s,.55,1.75,12.2,1.03,NAVY); text(s,.82,1.96,11.6,.6,'Working development system • supervised evaluation stage',27,True,WHITE)
for x,value,label in [(0.55,'96.3%','Damage-region recall\n78 / 81 test photos'),(4.72,'2.4%','Fresh clean false alerts\n11 / 456 photos'),(8.89,'5 classes','Dent, scratch, breakage,\npaint, deformation')]:
    rect(s,x,3.1,3.9,1.8,LIGHT); text(s,x+.22,3.28,3.5,.6,value,34,True,TEAL); text(s,x+.22,4.02,3.5,.7,label,16)
text(s,.65,5.35,11.9,1.05,'Prepared 06 Nov 2026. Latest recorded evaluation: 28 Oct 2026.\nProduction approval remains false; customer-photo performance is unmeasured.',19,False,MUTED)
finish(s,'Headline results reconstructed by merging E08, E09 and E10 by image ID, using glare grid 0.985|0.025|0.85 at threshold 0.381. These are region-alert metrics, not final API disposition or accuracy across all classes. No new model inference or training was performed for this deck.')

s=slide('What is built today','The end-to-end development workflow is implemented; several outputs remain unvalidated.','E01, E04–E07')
card(s,.55,1.8,3.9,3.9,'Detection + segmentation','RGB exterior photo → damage regions\nFive damage classes\nPinned v3 checkpoint + ONNX\nResolution and glare handling')
card(s,4.72,1.8,3.9,3.9,'Inspection report','Part/panel assignment\nDamage area per panel\nRule-based severity 1–5\nHeuristic view and side\nImage overlays + JSON')
card(s,8.89,1.8,3.9,3.9,'Application + safeguards','FastAPI + browser interface\nCLI and health endpoints\nReview / recapture routing\nArtifact integrity checks\nRollback selection retained')
text(s,.7,6.04,11.9,.55,'Progress stage: functional development baseline → data and validation work → production engineering',17,True,TEAL)
finish(s,'Code takes precedence over stale documentation: views.py and api.py implement heuristic vehicle side inference, despite an older readiness table saying it is not built. Its accuracy is not benchmarked. Presence of a feature is not evidence that its production gate passes.')

s=slide('How the model works','One DINOv2 backbone predicts damage presence, damage type and vehicle exterior.','E01, E04, E05; runs/ft_v3_mined_hardneg/profile.json')
stages=[('01  Capture quality','Check resolution, blur and exposure; poor inputs request recapture.'),('02  Prepare + tile','Long side 1,024–1,600 px; 768 px tiles, 192 px overlap.'),('03  Predict + combine','1.0× / 1.5× scales and flip TTA; combine probability maps.'),('04  Extract regions','Threshold 0.381; exterior floor 0.5; 1,200 px component guard.'),('05  Build the report','Glare to review; panel + side + severity; overlay and decision.')]
for i,(a,b) in enumerate(stages):
    y=1.72+i*.87; rect(s,.65,y,.52,.52,TEAL); text(s,.75,y+.05,.35,.4,str(i+1),16,True,WHITE); text(s,1.4,y,3.0,.5,a,18,True); text(s,4.55,y,7.9,.65,b,17)
text(s,.75,6.25,11.8,.42,'Backends: CUDA PyTorch or ONNX Runtime; Qualcomm laptop GPU supported. Phone deployment unproven.',15,False,MUTED)
finish(s,'TTA means test-time augmentation: predict the original and flipped tiles and combine their predictions. This is dense semantic segmentation with connected-region extraction, not an object detector alone. Working-resolution normalisation does not waive original-photo quality gates.')

s=slide('Measured progress across model iterations','Same public-photo evaluation groups; region alerts at threshold 0.381.','E02, E08–E12')
headers=['Configuration','Damage caught / 81','Fresh clean flagged / 456']
for x,w,h in [(0.65,4.9,headers[0]),(5.7,2.8,headers[1]),(8.7,3.9,headers[2])]: text(s,x,1.75,w,.5,h,17,True,TEAL)
data=[('v2 baseline','72  |  88.9%','64  |  14.0%'),('v3 hard-negative training','71  |  87.7%','15  |  3.3%'),('+ working-resolution normalisation','78  |  96.3%','15  |  3.3%'),('+ conservative glare filter  [current]','78  |  96.3%','11  |  2.4%'),('v5 candidate  [rejected]','74  |  91.4%','33  |  7.2%')]
for i,row in enumerate(data):
    y=2.4+i*.66; rect(s,.55,y,12.2,.59,'E4F2F0' if i==3 else LIGHT)
    for x,w,t in [(0.7,4.9,row[0]),(5.72,2.8,row[1]),(8.72,3.8,row[2])]: text(s,x,y+.11,w,.45,t,16,i==3)
text(s,.7,5.96,11.8,.65,'Main gains: fewer clean-car false alerts from hard negatives; better scratch visibility from resizing.\nThe glare settings used evaluation photos, so these results need independent confirmation.',16,False,MUTED)
finish(s,'Fresh-clean and held-out clean groups are distinct. v3 trained on 537 negatives mined from 1,674 clean photos. v5 improved a proxy but regressed in the real pipeline and was rejected. Table combines source-recorded historical values with reconstructed current counts. No untouched-test claim is made after tuning on evaluation images.')

s=slide('What the current numbers prove—and leave open','Small public-photo sets support development progress; they do not establish customer-grade accuracy.','E02, E03, E08–E11')
card(s,.55,1.8,3.9,3.45,'Damage-region recall','78 / 81 = 96.3%\n95% interval ≈ 90–99%\n3 damage cases missed\nBelow the ≥97% point target',TEAL)
card(s,4.72,1.8,3.9,3.45,'Clean-photo false alerts','Fresh: 11 / 456 = 2.4%\n95% interval ≈ 1.3–4.3%\nTest: 0 / 122; upper ≈ 3.1%\nPublic data; no customer test',TEAL)
card(s,8.89,1.8,3.9,3.45,'Protocol limits','Case score = surviving region\nNot final API decision recall\nNo current per-class gate\nReview / recapture rates unmeasured',AMBER)
text(s,.7,5.62,11.8,.9,'The next benchmark must measure all four API outcomes, class-level recall, false alerts,\nreview workload and vehicle-level leakage on a locked first-party test set.',19,True,INK)
finish(s,'Intervals are source-rounded 95% Wilson intervals. 0/122 does not demonstrate zero underlying false-positive risk. The threshold was retained from v2 and the glare grid was chosen with evaluation photos. Two user scratches were region-detected but quality metadata says low_resolution, so that does not demonstrate autonomous acceptance.')

s=slide('A real failure case: clean vehicle with an LED strip','The saved example exposes a false damage region and shows why final routing matters.','E13; pipeline_steps/white_scorpio_led/')
picture(s,'docs/assets/steps/step_01_original.jpg',.6,1.72,3.5,3.9)
picture(s,'docs/assets/steps/step_07_regions_on_photo.jpg',4.18,1.72,3.5,3.9)
picture(s,'docs/assets/steps/step_12_annotated_final.jpg',7.76,1.72,3.5,3.9)
for x,label in [(.7,'Original clean vehicle'),(4.28,'LED region predicted as damage'),(7.86,'Saved final overlay')]: text(s,x,5.75,3.5,.65,label,16,True)
text(s,.7,6.35,11.8,.4,'Saved final decision: recapture_required (399 × 501 px). A visible region is not an approved damage verdict.',15,True,AMBER)
finish(s,'Images are actual repository artifacts, not generated illustrations. E13 contains damage_present true and a crack_or_breakage region assigned to front_bumper, but decision recapture_required, automated_decision_allowed false and quality:low_resolution. Readiness record says more aggressive LED filtering lost 6–8 real damage cases, hence conservative routing.')

s=slide('What the model still lacks','Data quality and independent validation are the largest blockers.','E02, E03, E06, E07')
items=[('Damage type','Historical v2 precision: scratch 32%, paint 36%; current v3 unverified.'),('Damage boundaries','Proxy mask IoU ≈0.38; required mask mAP@50 has not been measured.'),('Dent depth + severity','No depth model; dents assume d2. Severity not checked against assessors.'),('Panel + side accuracy','Parts mIoU 0.62; weak mirrors/windows. Side rules exist, unbenchmarked.'),('Customer coverage','No independent customer-phone benchmark; subtle damage needs its own slices.'),('Service + deployment','No auth or monitoring. API 13–24 s reported; mobile parity not demonstrated.')]
for i,(a,b) in enumerate(items):
    y=1.73+i*.78; text(s,.7,y,3.0,.55,a,18,True,AMBER); text(s,3.75,y,8.8,.64,b,17)
finish(s,'Type figures are historical v2 pixel diagnostics inside predicted regions, not current v3 instance precision or vehicle-level accuracy. IoU and mAP are distinct metrics; the gate needs direct measurement. Parts and latency values are recorded diagnostics, not new benchmarks. License clearance, retention and rollback drills are also required in the workbook.')

s=slide('Production roadmap | four workstreams','Sequence the work around evidence gates; the workbook contains the detailed task register.','Proposed plan; based on E02–E07')
card(s,.55,1.82,2.9,3.95,'01  Data foundation','Agree labels + capture rules\nCollect first-party photos\nMask and audit annotations\nSplit by vehicle / session\nReview data rights',body_size=16)
card(s,3.66,1.82,2.9,3.95,'02  Model quality','Train balanced classes\nMine clean reflections\nImprove small-damage masks\nCalibrate on separate data\nValidate panels + severity',body_size=16)
card(s,6.77,1.82,2.9,3.95,'03  Product + runtime','Measure full API latency\nOptimize chosen hardware\nAdd auth, quotas and logs\nBuild reviewer workflow\nVerify exports + rollback',body_size=16)
card(s,9.88,1.82,2.9,3.95,'04  Independent release','Freeze model and policy\nRun locked external test\nCheck every critical slice\nRun supervised pilot\nCanary with rollback',body_size=16)
text(s,.7,6.07,11.8,.55,'Suggested owners: product lead, CV/ML engineer, data/annotation lead, QA, backend/MLOps and assessor.',16,False,MUTED)
finish(s,'This is a proposed execution plan, not completed work. Workbook weeks are relative planning windows and assume staffing, data access and parallel engineering. Passing release gates, rather than reaching a calendar week, controls release. Mobile export is conditional on choosing on-device deployment.')

s=slide('What must pass before production','Existing project targets plus proposed gates that require product and engineering agreement.','E02 for existing targets; all additional gates are proposals')
gate_rows=[('Detection','Recall ≥97%; clean false alerts <5%','Existing project targets'),('Type + localization','Per-class precision / recall ≥90%; mask mAP@50 >0.65','Type target proposed; mask target existing'),('Panels + severity','Panel mIoU ≥0.75; side accuracy ≥95%; severity κ ≥0.80','Proposed; include coverage / unknown rate'),('Full API performance','p95 end-to-end <10 s on chosen hardware','Proposed percentile for existing latency target'),('Review + operations','Report abstention; auth, retention, monitoring and rollback tested','Proposed operational acceptance'),('Independent evidence','Locked customer test; vehicle-separated splits; 95% CIs','Required validation; sample size plan first')]
for i,(a,b,c) in enumerate(gate_rows):
    y=1.73+i*.78; rect(s,.6,y,12.1,.71,LIGHT); text(s,.77,y+.08,2.6,.48,a,16,True,TEAL); text(s,3.4,y+.04,9.0,.38,b,16); text(s,3.4,y+.41,9.0,.26,c,10,False,MUTED)
finish(s,'Proposed numbers are planning defaults, not industry standards or user-approved SLAs. Workbook distinguishes existing and proposed targets and defines measurement units. Case-score recall must be supplemented by final automated decision performance and coverage to prevent low-quality/review cases from being counted as automatic successes. Dent depth must be validated or kept as an explicit assumption with assessor review.')

s=slide('Recommended next milestone','Prove performance on the intended customer workflow before approving autonomous decisions.','Proposed next actions; workbook task IDs T01–T36')
card(s,.65,1.85,7.5,3.9,'Immediate priorities','1   Agree capture protocol, labels and release metrics.\n2   Secure and label first-party clean + damaged vehicles.\n3   Freeze independent calibration and test groups.\n4   Retrain and compare in the complete API pipeline.\n5   Pilot with assessors and close production gates.',body_size=19)
card(s,8.4,1.85,4.25,3.9,'Deliverable to review','A reproducible candidate + locked-test report + operational release checklist.\n\nSee the Excel roadmap for dependencies, owners, proposed weeks and evidence.',body_size=18)
text(s,.8,6.11,11.6,.55,'Current recommendation: supervised evaluation with review of every result; retain development status.',18,True,AMBER)
finish(s,'No overall percent complete is assigned: implementation is substantial but production evidence is missing. The workbook is an editable plan, not a record of completed future tasks. Full source index and metric caveats are included there.')

ppt=OUT/'Exterior_Model_Progress.pptx'; prs.save(ppt)
thumbs=[]
for i,im in enumerate(previews):
    im.save(OUT/f'slide_{i+1:02}.png')
    t=im.resize((640,360)); thumbs.append(t)
sheet=Image.new('RGB',(1280,1800),'#D8E1E8')
for i,t in enumerate(thumbs):sheet.paste(t,((i%2)*640,(i//2)*360))
sheet.save(OUT/'Deck_Preview.png')

# Task register: IDs, sequencing, ownership, planning windows and explicit acceptance.
tasks=[
('T01','Define','P0','Agree scope and release decisions','Product lead','Triage versus automation; supported damage classes, panels and capture conditions.','Signed scope and metric definitions; repair cost and hidden damage excluded.','',1,1,'Not started','No production scope or approved complete gate set evidenced.'),
('T02','Define','P0','Freeze annotation taxonomy','CV lead + assessor','Define dent, scratch, paint, crack/breakage, detachment and overlaps; include tiny damage.','Annotation handbook with edge cases; 50-image adjudicated reference set.','T01',1,2,'Not started','Broken-part polygons can cover scuffs; historical type confusion.'),
('T03','Define','P0','Review rights and image handling','Data governance + legal reviewer','Audit source licenses and intended commercial use; define consent, access, retention and deletion.','Approved data-rights register and handling policy; no unresolved source used in release training.','T01',1,3,'Not started','Provenance manifests exist; clearance not evidenced. No legal conclusion assumed.'),
('T04','Data','P0','Specify guided capture protocol','Product + assessor','Required viewpoints, closeups, scale/distance, focus, lighting, occlusion and retake instructions.','Protocol tested on target phones; blur/glare and missing views clearly routed.','T01,T02',1,2,'Not started','Quality gate exists; real capture protocol not validated.'),
('T05','Data','P0','Design independent evaluation sample','QA / statistician','Choose class and nuisance slices, independent vehicle counts and confidence interval precision.','Approved sample-size plan; initial 100 clean + 100 damaged is pilot only, not release proof.','T01,T02',1,2,'Not started','81 damage cases produce a wide CI; no first-party benchmark.'),
('T06','Data','P0','Collect first-party exterior captures','Data lead','Recruit varied makes, colors, phones, sites and weather; include clean reflections and subtle damage.','Consented pilot corpus with >=100 clean and >=100 damaged vehicles; expand to T05 plan.','T03,T04,T05',2,5,'Not started','Existing evaluation primarily public photos.'),
('T07','Data','P0','Create vehicle-separated splits','Data engineer + QA','Group by vehicle, session and near duplicate; reserve train, validation, calibration and locked test.','Zero known vehicle/session/duplicate leakage; immutable split manifests and hashes.','T06',3,5,'Not started','Fresh independent splits required after tuning on older evaluation photos.'),
('T08','Data','P0','Annotate masks, panels and conditions','Annotation lead + assessors','Pixel masks, per-region class, panel, side, size, nuisance and clean status; retain ambiguity.','All pilot images labeled; ambiguous labels adjudicated; class counts published.','T02,T06',3,6,'Not started','About 570 original labeled damage images; masks/type coverage are limiting.'),
('T09','Data','P0','Audit annotation consistency','QA + assessor','Double-label a stratified 10% sample, minimum 50 images; audit fine scratches separately.','Proposed: inter-rater mask IoU >=0.80 and class agreement >=90%; resolve discrepancies.','T08',4,6,'Not started','Target proposed; meaningful consistency evidence missing.'),
('T10','Baseline','P0','Benchmark current full API','ML engineer + QA','Replay all inputs with final decision, quality, type, mask, panel, timing and model hash.','Reproducible baseline with all four outcomes; denominators and conditional automation coverage explicit.','T07,T09',5,6,'Not started','Saved region-alert metrics exist; final decision benchmark absent.'),
('T11','Training','P0','Expand class-balanced damage training','CV engineer','Add thin scratches, shallow dents, paint loss and true cracks; correct broad bumper labels.','Training split augmented; per-class instance/pixel counts and sampling documented.','T07,T09',5,8,'Not started','Current v3 type metrics unverified; historical scratch/paint precision weak.'),
('T12','Training','P0','Mine hard clean examples','CV engineer','Use training pool only: LED strips, chrome, seams, reflections, dirt, showroom and repaired paint.','Mined negatives independently checked; no calibration/test mining or contamination.','T07,T10',6,8,'Partially implemented','v3 already uses 537 mined hard negatives; customer-domain extension needed.'),
('T13','Training','P0','Improve segmentation boundaries','CV engineer','Compare higher-resolution features/decoder, class-aware losses and tiny-damage sampling.','Validation mask metrics and small-defect recall improve without unacceptable false alerts.','T11,T12',7,10,'Not started','Mask IoU approximately 0.38 on proxy; mask mAP@50 unmeasured.'),
('T14','Panels','P1','Improve and benchmark parts','CV engineer','Target mirrors, windows, bumpers and occlusion; assess downstream panel damage area.','Proposed: panel mIoU >=0.75 overall; critical-part results and area error published.','T08,T09',6,10,'Partially implemented','Part model exists: reported mIoU 0.62; mirror 0.28 and front window 0.23.'),
('T15','Panels','P1','Validate vehicle view and side','CV engineer + QA','Benchmark existing heuristics; retain unknown on ambiguous captures; consider dedicated model if needed.','Proposed: >=95% side accuracy on assigned sides with >=90% eligible-case coverage.','T08,T14',8,10,'Partially implemented','Current views.py implements heuristics; accuracy and unknown rate unmeasured.'),
('T16','Severity','P1','Resolve dent-depth requirement','Product + assessor + CV lead','Choose reviewed assumed-depth triage or collect depth labels with multi-view/measurement reference.','Approved depth policy; remove implicit depth claims; define depth model gate if automated rating required.','T01,T08',5,7,'Not started','Current dent rating assumes d2; no depth model.'),
('T17','Severity','P1','Validate damage severity rubric','Assessor + QA','Rate >=100 panels for pilot, expanded per sample plan; compare all types and area bins.','Proposed weighted kappa >=0.80 and >=90% within one level; validate breakage/paint extensions.','T14,T16',9,11,'Partially implemented','Rule-based 1–5 exists; no human-agreement benchmark.'),
('T18','Calibration','P0','Calibrate complete decision policy','ML engineer','Select thresholds, component guard, glare and uncertainty on calibration split only.','Freeze full policy and hashes; report recall, false alerts and abstention before test access.','T10,T13',10,11,'Not started','0.381 retained from proxy/v2; glare settings tuned using evaluation photos.'),
('T19','Evaluation','P0','Run independent locked test','QA owner independent of training','Evaluate final frozen candidate across all damage classes and clean vehicles.','Recall >=97%, clean false alerts <5%; proposed class P/R >=90%, with 95% CIs and gate policy.','T18,T23,T27,T29',12,13,'Not started','Latest public region metrics: 78/81 damage; 0/122 test clean; 11/456 fresh clean.'),
('T20','Evaluation','P0','Verify nuisance and minor-damage slices','QA + assessor','Check dark/metallic paint, LEDs, night, dirt, wet panels, occlusion, tiny scratches, glass and repairs.','Each agreed critical slice passes predeclared gate; failures block or explicitly restrict scope.','T19',13,14,'Not started','Two user scratches are anecdotes; LED false region remains.'),
('T21','Runtime','P0','Select deployment target and SLO','Backend lead + product','Choose server/GPU versus phone; specify concurrency, cost and full response latency.','Written hardware, p95 latency, throughput, availability and cost budget.','T01',1,2,'Not started','Laptop Qualcomm ONNX demonstrated; mobile unproven.'),
('T22','Runtime','P0','Profile full API path','ML + backend engineer','Measure decoding, tiling, model, parts, overlays and response serialization under load.','Stage timing and p50/p95/p99 for agreed image sizes and concurrency.','T10,T21',6,7,'Not started','Reported 8.1 s model average; 13–24 s API; CPU 60–280 s.'),
('T23','Runtime','P0','Optimize and parity-test selected backend','ML systems engineer','Profile tile/TTA reduction, batching, precision and report rendering; recalibrate if output changes.','Proposed p95 API <10 s; pixel/region/API parity tolerances agreed and passed.','T13,T21,T22',8,11,'Partially implemented','ONNX + GPU support exists; end-to-end production load target unverified.'),
('T24','Runtime','P1','Validate mobile export if required','Mobile + ML systems engineer','Conditional task: choose mobile runtime after T21; measure memory, battery, latency and output parity.','If on-device selected: approved parity >0.98 plus device-matrix and sustained-runtime report.','T21,T23',11,14,'Conditional','Existing project mobile parity target; current laptop runtime is not phone proof.'),
('T25','Service','P0','Add auth, quotas and input controls','Backend engineer','Authenticate callers; enforce rate/size/pixel limits, timeouts and safe image decoding.','Unauthorized/oversized/malformed requests rejected; abuse and timeout checks pass.','T21',3,6,'Partially implemented','API and byte-size limit exist; auth/rate limits absent.'),
('T26','Service','P0','Implement governed storage and audit trail','Backend + data governance','Apply access, encryption, retention/deletion and metadata-only logs; protect plate/person images.','Policy-backed deletion and audit tests; no raw photos in general application logs.','T03,T25',5,8,'Not started','Retention and operational controls not evidenced.'),
('T27','Service','P0','Build deployment and resilience controls','MLOps engineer','Container/packaging, CI, warmup, readiness, queues/backpressure, concurrency and failure recovery.','Reproducible deployment; agreed load, restart and out-of-memory recovery checks pass.','T21,T25',6,10,'Partially implemented','Health endpoints, warmup and integrity checks exist; production resilience absent.'),
('T28','Operations','P0','Monitor quality, review load and latency','MLOps + QA','Log immutable model/policy IDs, outcomes, confidence, timing, errors, slice drift and assessor corrections.','Dashboards and alerts tested; defined review/recapture and drift escalation policies.','T26,T27',8,11,'Not started','Current model status exists; production telemetry absent.'),
('T29','Workflow','P0','Implement assessor review and recapture loop','Product + backend + assessor','Present evidence and uncertainty; capture corrections, retakes and escalation with ownership.','All four decisions handled; no unsupported automatic verdict; reviewer capacity established.','T04,T25',6,10,'Partially implemented','Fail-closed response and UI exist; operational workflow and capacity not demonstrated.'),
('T30','Release','P0','Version artifacts and rehearse rollback','MLOps + QA','Release manifest includes weights, calibration, parts, preprocessing, code and dataset hashes.','One-command rollback drill and backward-compatible API checks pass in staging.','T23,T27,T28',10,12,'Partially implemented','Pinned artifacts and previous selection retained; rollback drill missing.'),
('T31','Pilot','P0','Run supervised customer pilot','Operations + assessor + QA','Review every result; log false negatives, false alerts, retakes, time and override reasons.','Predeclared pilot sample and duration met; assessor agreement and incident report approved.','T17,T19,T20,T28,T29,T30',14,16,'Not started','Current recommendation is supervised evaluation only.'),
('T32','Release','P0','Approve evidence and release scope','Product + ML lead + operations','Review locked test, class/slice gates, rights, latency, severity and incidents.','All applicable gates pass; accountable owner signs release and unsupported use restrictions.','T03,T15,T17,T20,T23,T30,T31',16,16,'Not started','production_approved remains false.'),
('T33','Release','P0','Canary production rollout','MLOps + operations','Stage traffic, retain human review where needed and monitor approved rollback triggers.','Canary meets agreed error, recall-audit and latency budgets; rollback tested and staffed.','T32',17,18,'Not started','No production rollout evidenced.'),
('T34','Operations','P1','Maintain feedback and retraining cycle','ML + data lead','Label reviewed failures, version fresh data and recalibrate candidates without reusing locked test.','Periodic drift review; new candidate passes fresh independent release gates before promotion.','T28,T33',18,20,'Not started','Ongoing activity after release; the scheduled window covers initial setup, not completion of maintenance.'),
('T35','Evaluation','P0','Set confidence-interval release policy','QA / statistician + product','Choose whether recall/FPR gates apply to point estimates or CI bounds and power critical slices.','Predeclare CI method, unit and bound policy; resize test set before any test inspection.','T05',2,3,'Not started','Suggested conservative release: recall lower CI >=97%, FPR upper CI <5%; feasibility requires sample design.'),
('T36','Evaluation','P0','Audit abstention and automated decisions','QA + product','Measure final damage verdict precision/recall, eligible coverage, review and recapture rates.','Agreed workflow gates pass; region recall never substituted for autonomous-decision performance.','T18,T29,T35',11,12,'Not started','Saved score reports do not evaluate complete final API outcomes.'),
]
# Enforce dependencies that are critical to an interpretable locked test.
tasks=[tuple(list(t[:7])+[t[7]+',T35,T36']+list(t[8:])) if t[0]=='T19' else t for t in tasks]
# Preserve estimated durations while moving downstream work past prerequisites.
for _ in range(len(tasks)):
    changed=False; lookup={t[0]:t for t in tasks}; revised=[]
    for t in tasks:
        deps=list(filter(None,t[7].split(',')))
        earliest=max([t[8]]+[lookup[d][9] for d in deps])
        if earliest>t[8]:
            shifted=list(t); shifted[8]=earliest; shifted[9]=earliest+t[9]-t[8]; t=tuple(shifted); changed=True
        revised.append(t)
    tasks=revised
    if not changed: break
else: raise ValueError('Dependency cycle in task plan')
horizon=max(t[9] for t in tasks)

wb=xlsxwriter.Workbook(OUT/'Exterior_Model_Production_Roadmap.xlsx')
wb.set_properties({'title':'Exterior damage model: production roadmap','subject':'Evidence-backed current progress and proposed release plan','author':'Project review','comments':'Prepared 2026-11-06 from local repository evidence; no new inference run.'})
fmt_title=wb.add_format({'bold':True,'font_size':22,'font_color':NAVY})
fmt_sub=wb.add_format({'font_size':11,'font_color':MUTED,'text_wrap':True,'valign':'top'})
fmt_header=wb.add_format({'bold':True,'bg_color':NAVY,'font_color':WHITE,'text_wrap':True,'valign':'vcenter'})
fmt_body=wb.add_format({'font_size':11,'text_wrap':True,'valign':'top'})
fmt_int=wb.add_format({'num_format':'0','valign':'top'})
fmt_pct=wb.add_format({'num_format':'0.0%','valign':'top'})
fmt_link=wb.add_format({'font_color':'007F82','underline':1,'text_wrap':True,'valign':'top'})
fmt_green=wb.add_format({'bg_color':'E1F0E6','font_color':GREEN})
fmt_amber=wb.add_format({'bg_color':'FFF0D8','font_color':AMBER})
fmt_red=wb.add_format({'bg_color':'F9E4E4','font_color':RED})
def init(name,title,sub,widths):
    ws=wb.add_worksheet(name); ws.hide_gridlines(2); ws.set_tab_color(TEAL); ws.set_default_row(42)
    ws.merge_range(0,0,0,len(widths)-1,title,fmt_title); ws.set_row(0,34)
    ws.merge_range(1,0,2,len(widths)-1,sub,fmt_sub); ws.set_row(1,25); ws.set_row(2,25)
    for i,width in enumerate(widths):ws.set_column(i,i,width,fmt_body)
    ws.freeze_panes(5,1); ws.set_landscape(); ws.fit_to_pages(1,0); ws.repeat_rows(0,4); ws.set_paper(9)
    ws.set_footer('&LExterior damage model | 06 Nov 2026&RPage &P of &N')
    return ws
def table(ws,headers,data):
    ws.add_table(4,0,4+len(data),len(headers)-1,{'style':'Table Style Medium 2','columns':[{'header':h} for h in headers],'data':data})
    ws.set_row(4,32)
    for r in range(5,5+len(data)):ws.set_row(r,78)

ws=init('Read Me','Exterior model | production roadmap','Prepared 06 Nov 2026. Latest recorded evaluation: 28 Oct 2026. Proposed owners, weeks and additional targets require team agreement; no new training or model inference was run.',[28,105])
guide=[('Current verdict','Development only; supervised evaluation recommended. production_approved is false. No defensible overall completion percentage can be inferred.'),('What is implemented','DINOv2 v3 + ONNX; full API/UI; quality and glare routing; parts; heuristic side; rule severity; pinned artifacts and rollback selection.'),('Verified headline counts','78/81 damaged-region alerts; 0/122 test-clean alerts; 11/456 fresh-clean alerts at threshold 0.381. Reconciled from saved score files E08–E10.'),('Metric meaning','Region-survival case scores, not final API damage decisions. Review and recapture outcomes must be audited separately. Zero alerts on 122 images does not prove zero underlying FPR.'),('Evidence caveat','Historical v2 scratch/paint and proxy mask diagnostics do not establish current v3 performance. The glare grid used evaluation photos; all current figures need independent first-party confirmation.'),('How to use','Assign named owners, change task Status and Week start/end, agree Acceptance Gates, then track evidence links. Task summary and Gantt update from the task register.'),('Timeline assumptions','Week numbers are relative to a future approved kickoff, not promised dates. Windows assume parallel data and engineering work. Staff capacity, data access and labeling rates are unknown.'),('Status meanings','Not started = no completion evidence; Partially implemented = existing feature needs production work; Conditional = needed only if selected architecture/deployment requires it; Complete = acceptance evidence approved.'),('Release policy','Lock weights, preprocessing and thresholds before independent testing. Failed locked-test gates require a new candidate and fresh evaluation policy; do not tune repeatedly on the locked test.'),('Data volume','100 clean + 100 damaged vehicles is a starting pilot collection only. Statistician must size independent cases and class/slice coverage before release testing.'),('Mobile and depth','Mobile export is conditional on deployment choice. Dent-depth policy must be resolved; assumed d2 cannot be presented as measured depth. Apply assessor review until validated.'),('Source precedence','Current code resolves implemented behavior; selected manifest resolves checkpoint; newest readiness update resolves recorded metrics; raw score files support count reconstruction. See Evidence Index.')]
table(ws,['Topic','Guidance'],guide)

ws=init('Task Register','Detailed production task register','Editable suggested plan. Dependencies are task IDs; owner roles are suggestions. Week 1 starts when the team authorizes kickoff. Partial implementation is not a passed release gate.',[9,15,9,37,29,64,67,27,12,12,23,65])
headers=['ID','Workstream','Priority','Action','Suggested owner','Implementation / deliverable','Acceptance evidence','Dependencies','Week start','Week end','Status','Current basis / gap']
table(ws,headers,tasks)
ws.data_validation(5,10,4+len(tasks),10,{'validate':'list','source':['Not started','Partially implemented','Conditional','In progress','Blocked','Complete']})
ws.data_validation(5,8,4+len(tasks),9,{'validate':'integer','criteria':'between','minimum':1,'maximum':104})
ws.conditional_format(5,10,4+len(tasks),10,{'type':'text','criteria':'containing','value':'Complete','format':fmt_green})
ws.conditional_format(5,10,4+len(tasks),10,{'type':'text','criteria':'containing','value':'Partially','format':fmt_amber})
ws.conditional_format(5,2,4+len(tasks),2,{'type':'text','criteria':'containing','value':'P0','format':fmt_red})

gates=[
('G01','Case detection recall','Existing project target','>=97% damage-region recall; also report final API recall','78/81 = 96.3%; 95% CI about 90–99%','Below target / public data only','Independent damaged vehicles; full API plus comparable region metric','QA + ML lead','T18,T19,T35,T36','E02,E08–E11'),
('G02','Clean false-alert rate','Existing project target','<5%; predeclare CI-bound policy','11/456 fresh = 2.4%; 0/122 test; no first-party measurement','Public point target met; customer gate open','Independent clean vehicles with ground-truth review; all nuisance slices','QA + product','T19,T20,T35','E02,E08–E10'),
('G03','Damage class precision / recall','Proposed','>=90% precision and recall for every required class','Historical v2 pixel precision: scratch 32%, paint 36%, crack 87%, deformation 73%','Current v3 not measured','Instance matching policy + per-class confusion matrix and 95% CIs; include dent','CV + QA','T02,T11,T19','E03'),
('G04','Mask localization','Existing project target','mask mAP@50 >0.65','Proxy mean IoU ~0.38; mAP not measured','Open; IoU is not mAP','Define semantic-to-instance conversion, matching, size bins; compute mask AP directly','CV + QA','T13,T19','E02,E03'),
('G05','Parts / panel assignment','Proposed','mIoU >=0.75; per-part and damage-area errors disclosed','Parts mIoU 0.62; mirror 0.28; front window 0.23','Partial','Held-out part masks and damage-to-panel attribution accuracy','CV + QA','T14','E02'),
('G06','View / side','Proposed','>=95% assigned-side accuracy; >=90% eligible-case coverage','Heuristics implemented, accuracy unmeasured','Open','Vehicle-centric side ground truth; unknown rate and errors by view','CV + QA','T15','E06'),
('G07','Dent depth','Proposed policy gate','Measured-depth model validated, or explicit assumed-depth workflow with human review','No depth model; d2 assumed','Open','Depth-reference labels if required; otherwise signed restricted-use policy','Product + assessor','T16','E07'),
('G08','Severity agreement','Proposed','Weighted kappa >=0.80; >=90% within one level','1–5 rubric implemented; no assessor benchmark','Open','>=100 panels initially; expand by T05 plan; depth/area/type strata','Assessor + QA','T17','E07'),
('G09','End-to-end latency','Existing target; proposed p95 definition','p95 <10 s on agreed hardware / image sizes / concurrency','Reported model avg 8.1 s; API 13–24 s; CPU 60–280 s','Full API gate not met / p95 unmeasured','Cold/warm latency and load test incl. parts, report, transfer and serialization','ML systems + backend','T21–T23','E02'),
('G10','Mobile parity','Existing project target; conditional','Parity >0.98 plus agreed latency/memory/battery','Laptop ONNX/QNN only; no mobile demonstration','Conditional','Define parity metric; device-matrix and sustained-run tests','Mobile + ML systems','T24','E02'),
('G11','API outcomes and review burden','Proposed','Agree final-decision precision/recall and review/recapture coverage before test','Four routes exist; workload/coverage not measured','Open','Confusion by final route; rate and reason for review/recapture; assessor capacity','Product + QA','T29,T36','E04,E11'),
('G12','Security / retention / operations','Proposed','All auth, quotas, retention, monitoring and recovery checks pass','Integrity and health present; auth/monitoring absent','Open','Staging evidence and accountable owners; failure injection and deletion tests','Backend + MLOps','T25–T30','E02,E04'),
('G13','Data rights and provenance','Project blocker; proposed closure','Source and commercial-use review completed; provenance and consent maintained','Manifests exist; rights review unresolved','Open','Qualified reviewer signs source register and intended-use controls','Data governance','T03','E02,E14'),
('G14','Independent release evidence','Proposed','Predeclared confidence policy and slice gates; frozen model/policy; independent signoff','Public-photo tuning; no untouched customer test','Open','T35 decides point vs CI bounds; proposed conservative CI lower recall >=97%, upper FPR <5%','QA + product','T05,T07,T19,T20,T31,T32','E02,E10'),
]
ws=init('Acceptance Gates','Production acceptance gates','Existing targets come from PRODUCTION_READINESS.md. Proposed values are planning defaults, not standards or approved SLAs. Confidence-bound policy must be agreed and sample-sized before test access.',[9,27,27,63,67,35,73,28,23,20])
table(ws,['Gate','Capability','Target origin','Release criterion','Current evidence','Status','How to validate','Accountable role','Related tasks','Evidence IDs'],gates)

data_plan=[
('Clean / undamaged','Reflections, chrome, LEDs, seams, showroom, outdoors, dirt and wet paint','Verified clean status; vehicle/session IDs; nuisance tags','Pilot >=100 independent clean vehicles; expand via T05','False alert rate and CI; separate pristine vs wear','T06–T09'),
('Dent','Shallow and deep, small and large; different panels and paints','Pixel masks, panel, side; depth reference only if required','Per-class positive count sized for agreed CI; avoid image-count-only quotas','Dent recall, mask AP and severity; subtle-dent slice','T08,T11,T16'),
('Scratch','Fine/light scratches, long marks, scuffs, dark/red/white/silver paint','Thin masks; class distinction from cracks and paint loss','Oversample rare subtle scratches in training; locked natural-prevalence test plus challenge set','Per-class P/R, small-mask AP and capture-resolution slice','T08,T11,T13'),
('Crack / breakage','True fractures, glass/lamp breakage, torn bumper, detached pieces','Tight region masks; separate scuff from broken part','Independent vehicles across severity and parts','Precision/recall; distinguish cracks from scratch and paint','T02,T08,T11'),
('Paint damage','Peeling, chips, scraped paint; color and lighting variation','Masks and class boundary; assessor adjudication','Add verified paint examples; quantify current class imbalance','Paint precision/recall; scratch confusion','T02,T08,T11'),
('Deformation / detachment','Bent panels, displaced trim/bumper, missing exterior pieces','Class, mask, affected part and visibility','Represent real failure mechanisms; clarify missing-part label policy','Per-class P/R and panel assignment','T02,T08,T11'),
('Quality / recapture','Blur, low resolution, over/underexposure, glare and occlusion','Capture acceptability and retake reason; preserve original resolution','Deliberately collect bad inputs for routing tests','Retake precision/recall, automation coverage and review load','T04,T29,T36'),
('Panels + views','Front/rear/side/three-quarter; mirrors, windows and bumpers','Part mask; vehicle-centric side; unknown eligibility','Coverage over required parts and all supported views','Part mIoU, side accuracy and unknown coverage','T14,T15'),
('Severity','All types, area bins and d1/d2/d3 where depth evidence exists','Assessor 1–5 scores; rubric basis; multiple raters','>=100 panels for pilot; expand to approved precision','Weighted kappa, within-one-level and disagreement review','T16,T17'),
('Split / leakage controls','Vehicle, repeated session, dealer/site, source URL and near duplicates','Group IDs, hashes, timestamps and immutable manifest','Train / validation / calibration / locked test separated by vehicle','Leakage audit; independent test cannot become hard-negative pool','T05,T07,T35'),
]
ws=init('Data Plan','Data and annotation collection plan','Pilot collection starts at >=100 clean + >=100 damaged independent vehicles in total. This is not a statistically sufficient release sample by itself; T05/T35 determine final per-class and slice counts.',[27,69,67,66,66,22])
table(ws,['Slice','Capture coverage','Labels / metadata','Suggested collection rule','Evaluation evidence','Task links'],data_plan)

ws=init('Current Evidence','Current progress and evidence ledger','Recorded measurements, implemented features and unmeasured items are kept distinct. Headline counts reconstructed from saved files; no new inference or test-suite run is represented here.',[32,67,54,57,24])
metrics=[('Current selected model','ft_v3_mined_hardneg; threshold 0.381','Implemented; development only','Resolution normalization + conservative glare routing','E01,E02,E04,E05'),('Damage-region recall','78/81 = 96.3%; 95% CI ~90–99%','Measured public-photo case score','Not final API decision recall; post-tuning optimistic','E08–E11'),('Test-clean alerts','0/122 = 0%; 95% CI upper ~3.1%','Measured public-photo case score','Does not prove true FPR is zero','E08–E10'),('Fresh-clean alerts','11/456 = 2.4%; 95% CI ~1.3–4.3%','Measured public-photo case score','Glare grid used evaluation photos','E08–E10'),('Historical type diagnostics','v2 scratch 32%, paint 36%, crack 87%, deformation 73% precision','Historical pixel-level diagnostic','Current v3 type precision unverified; not instance-level figures','E03'),('Mask quality','Mean IoU ~0.38 on proxy','Historical / proxy','Direct mask mAP@50 not measured','E02,E03'),('Part segmentation','mIoU 0.62; mirror 0.28; front window 0.23','Recorded part-model diagnostic','Benchmark attribution and panel-area error on first-party data','E01,E02'),('Vehicle side','Geometry-based heuristic; unknown/null supported','Implemented, unbenchmarked','Current code supersedes older table saying not built','E04,E06'),('Dent depth / severity','Depth assumed d2; severity rules 1–5','Implemented assumption; agreement unmeasured','Paint uses scratch rubric; crack/deformation use deepest-dent row','E07'),('Known LED false region','Clean Scorpio: damage region; final recapture_required','Saved example; not production success','399×501 low-resolution input; no automated damage verdict','E13'),('Latency','8.1 s model mean; API 13–24 s; CPU 60–280 s','Recorded laptop observations','Not p95 or a controlled load benchmark','E02'),('Artifacts and routing','Pinned weights/profile/ONNX/parts; review and recapture; rollback file','Implemented','No production rollback exercise or deployment evidence','E01,E04'),('Automated decisions','All four final routes exist','Feature implemented; outcome performance unmeasured','Measure route confusion, automation coverage and abstention','E04,E11'),('Customer validation','No independent first-party result found','Unmeasured','Public-photo performance is not customer validation','E02'),('v5 candidate','74/81 damage; 33/456 fresh clean alerts','Rejected experiment','Single-scale proxy improvement did not transfer','E02,E12')]
table(ws,['Item','Finding','Evidence status','Interpretation / remaining gap','Evidence IDs'],metrics)

ws=init('Timeline','Suggested execution windows | relative weeks','Editable weeks live in Task Register. Parallel work assumes suitable staffing; no fixed calendar delivery is promised. Release may move until all required evidence gates pass. Mobile task T24 is conditional.',[10,42]+[5]*horizon)
ws.write(4,0,'Task',fmt_header); ws.write(4,1,'Action',fmt_header)
for wk in range(1,horizon+1):ws.write(4,wk+1,wk,fmt_header)
for i,t in enumerate(tasks,5):
    ws.set_row(i,32); ws.write(i,0,t[0]);ws.write(i,1,t[3]); row=i+1
    for wk in range(1,horizon+1):
        formula=f'=IF(AND({wk}>=\'Task Register\'!I{row},{wk}<=\'Task Register\'!J{row}),1,0)'
        ws.write_formula(i,wk+1,formula,wb.add_format({'num_format':';;;'}),int(t[8]<=wk<=t[9]))
ws.conditional_format(5,2,4+len(tasks),horizon+1,{'type':'cell','criteria':'==','value':1,'format':wb.add_format({'bg_color':'168D91'})})
ws.autofilter(4,0,4+len(tasks),horizon+1)

ws=init('Summary','Task progress dashboard','Counts are task statuses, not a percentage of model quality or effort completed. Update Task Register statuses and the workbook recalculates. Partial features still need acceptance evidence.',[32,24,68])
table(ws,['Metric','Value','Meaning'],[
('Total planned tasks',len(tasks),'All detailed tasks, including conditional mobile packaging.'),
('Complete',0,'Acceptance evidence reviewed and accepted.'),('In progress',0,'Work actively underway.'),
('Partially implemented',sum(t[10]=='Partially implemented' for t in tasks),'Existing development features that require production work.'),
('Not started',sum(t[10]=='Not started' for t in tasks),'No implementation or completion evidence for the planned task.'),
('Conditional',sum(t[10]=='Conditional' for t in tasks),'Required only if deployment/scope decision makes it applicable.'),
('Blocked',0,'Task requires external input or dependency resolution.'),
('P0 tasks',sum(t[2]=='P0' for t in tasks),'Highest-priority release work; still scope-dependent.'),
('Current release status','Development only','Production approval is false; release signoff is task T32.'),
('Suggested initial horizon',f'{horizon} relative weeks','Illustrative windows only; data, staff capacity and quality gates control timing.')])
for r,status in [(6,'Complete'),(7,'In progress'),(8,'Partially implemented'),(9,'Not started'),(10,'Conditional'),(11,'Blocked')]:
    ws.write_formula(r,1,f'=COUNTIF(\'Task Register\'!K6:K41,"{status}")',fmt_int,sum(t[10]==status for t in tasks))
ws.write_formula(5,1,"=COUNTA('Task Register'!A6:A41)",fmt_int,len(tasks))
ws.write_formula(12,1,'=COUNTIF(\'Task Register\'!C6:C41,"P0")',fmt_int,sum(t[2]=='P0' for t in tasks))

ws=init('Evidence Index','Repository source index','All paths are relative to the project root. Links point back to local project files. The deck speaker notes explain protocol and historical-metric caveats.',[10,45,95,100])
table(ws,['ID','Source','Relative path','Use / caveat'],sources)
for i,(_,_,path,_) in enumerate(sources,5):
    ws.write_url(i,2,'external:../'+path,fmt_link,path)
wb.close()

# Validate artifact structures, task references and dependency windows.
ids={t[0] for t in tasks}; byid={t[0]:t for t in tasks}
for t in tasks:
    for dep in filter(None,t[7].split(',')):
        assert dep in ids,(t[0],dep)
        assert byid[dep][9]<=t[8],('Dependency ends after proposed start',t[0],dep)
for p in [ppt,OUT/'Exterior_Model_Production_Roadmap.xlsx']:
    with zipfile.ZipFile(p) as z: assert z.testzip() is None
assert len(Presentation(ppt).slides)==10
print(json.dumps({'files':[str(ppt),str(OUT/'Exterior_Model_Production_Roadmap.xlsx')],'verified_counts':verified,'slide_count':10,'tasks':len(tasks),'gates':len(gates),'workbook_sheets':8},indent=2))
