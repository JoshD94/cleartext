from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True);page=browser.new_page()
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((root/'outputs/semcor-model-review.html').as_uri())
    assert page.locator('details').count()==21
    for width in [1280,390]:
        page.set_viewport_size({'width':width,'height':900})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(root/f'work/semcor-review-{width}.png'),full_page=False)
    page.locator('summary').first.click()
    assert page.locator('details').first.get_attribute('open') is not None
    assert not errors
    browser.close()
print('Report verified at desktop and mobile widths; 20 traces plus bank probes.')
