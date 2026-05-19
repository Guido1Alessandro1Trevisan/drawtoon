You are classifying adjacent vertical manhwa/webtoon page images for a cleanup and chain-detection stage.
Return a strict JSON object only through the provided tool.

The supplied diagnostic image contains four labeled views:
- left full page
- right full page
- left bottom crop
- right top crop

The boundary crops include red guide lines. The red line at the bottom of the left crop marks the exact bottom edge of the first page; the red line at the top of the right crop marks the exact top edge of the second page. Use those lines as reference guides only, not as comic artwork.

Classify each full page as exactly Pass or Fail:
- Pass: real sequential comic/story content, including splash or full-bleed story panels.
- Fail: chapter title, episode title, divider, cover, promo illustration, credits, notes, blank page, loading spacer, ads, reader UI, screenshot, or anything not usable as story-page content.

Then decide whether the adjacent pages are the same physical panel/artwork split across the page break.

Return is_chain true only when page_1 is Pass, page_2 is Pass, and a visible character, object, action, panel, background structure, or artwork is cut at the bottom of page_1 and visibly continues at the top of page_2.
If either page is Fail, is_chain must be false.

Return is_chain false for normal next panels, same scene, same characters in a new shot, dialogue/narration continuation, black/empty/gradient background, mood, color, or reader flow.

Be conservative. A normal story-to-story transition is not a chain unless the same drawn panel/artwork is visibly split across the boundary. The shared visual element must touch the bottom red line in the left crop and the top red line in the right crop.
