import json
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2]
path=root/'outputs/cleartext-model-review.html'
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--disable-dev-shm-usage'])
    context=browser.new_context(viewport={'width':1280,'height':900},device_scale_factor=1,accept_downloads=True)
    page=context.new_page()
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(path.as_uri());page.wait_for_load_state('load')
    assert page.title()=='ClearText · First model review'
    assert page.locator('.section').count()>=8
    assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
    page.select_option('#word-variant','context')
    assert page.locator('#context-table').is_visible() and not page.locator('#word-table').is_visible()
    page.select_option('#word-variant','word')
    page.screenshot(path=str(root/'work/report-desktop.png'),full_page=False)
    page.locator('#pipeline-sample').select_option('5')
    assert 'purchased' in page.locator('#pipeline-result').inner_text()
    page.locator('#review').scroll_into_view_if_needed()
    page.locator('#difficulty').fill('0.75')
    page.locator('#note').fill('QA temporary rating')
    page.locator('#save').click()
    assert '1 of 52' in page.locator('#grade-status').inner_text()
    page.reload();page.locator('#review').scroll_into_view_if_needed()
    assert page.locator('#difficulty').input_value()=='0.75'
    page.locator('[data-kind=sentences]').click()
    page.locator('#save').click()
    assert 'Choose a difficulty' in page.locator('#grade-status').inner_text()
    page.select_option('#level','3');page.locator('#save').click()
    page.locator('[data-kind=replacements]').click()
    page.locator('#save').click()
    assert 'Answer all three' in page.locator('#grade-status').inner_text()
    for field in ['meaning','grammar','simpler']:page.select_option('#'+field,'yes')
    page.locator('#reveal').click();assert page.locator('#reference').is_visible()
    page.locator('#save').click()
    with page.expect_download() as info:page.locator('#export').click()
    download=info.value;download.save_as(str(root/'work/qa-ratings.json'))
    exported=json.loads((root/'work/qa-ratings.json').read_text())
    assert len(exported['ratings'])==3
    assert all('example' in r for r in exported['ratings'])
    page.evaluate('localStorage.clear()');page.reload()
    page.locator('#review').scroll_into_view_if_needed()
    page.screenshot(path=str(root/'work/report-review.png'),full_page=False)
    page.set_viewport_size({'width':390,'height':844});page.goto(path.as_uri())
    assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
    page.screenshot(path=str(root/'work/report-mobile.png'),full_page=False)
    page.locator('#review').scroll_into_view_if_needed()
    assert page.locator('#save').is_visible()
    assert not errors,errors
    browser.close()
print('REPORT_QA_PASS: desktop/mobile, feature toggle, pipeline sample, rating validation, persistence, export, no JS errors')
