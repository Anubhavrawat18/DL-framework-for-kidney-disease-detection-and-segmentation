param(
  [string]$Deck = 'C:\StudyMaterialDoNotTouch\publications\URI-CADS dataset\presentation\URI-CADS_Project_Review.pptx',
  [string]$Script = 'C:\StudyMaterialDoNotTouch\publications\URI-CADS dataset\presentation\Presentation_Script_Two_Presenters.docx',
  [string]$OutDir = 'C:\Users\anubh\AppData\Local\Temp\claude\C--StudyMaterialDoNotTouch-publications-URI-CADS-dataset\f3207856-96cc-4463-a75d-ed0cec1b22d9\scratchpad\qa',
  [string]$Slides = ''
)
New-Item -ItemType Directory -Force $OutDir | Out-Null

# Attach to PowerPoint without disturbing a deck the user already has open: open ours read-only and
# windowless, export, close only ours, and quit only if this script started the application.
$pptWasRunning = [bool](Get-Process POWERPNT -ErrorAction SilentlyContinue)
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($Deck, -1, 0, 0)
$wanted = if ($Slides) { $Slides.Split(',') | ForEach-Object { [int]$_ } } else { 1..$pres.Slides.Count }
foreach ($i in $wanted) {
  $pres.Slides.Item($i).Export((Join-Path $OutDir ('slide-{0:D2}.png' -f $i)), 'PNG', 1600, 900)
}
"Exported $($wanted.Count) of $($pres.Slides.Count) slides"
$pres.Close()
if (-not $pptWasRunning) { $ppt.Quit() }
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null

if (-not $Slides) {
  $wordWasRunning = [bool](Get-Process WINWORD -ErrorAction SilentlyContinue)
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $doc = $word.Documents.Open($Script, $false, $true)
  $pdf = Join-Path $OutDir 'script.pdf'
  $doc.SaveAs2($pdf, 17)
  "Script pages: $($doc.ComputeStatistics(2))"
  $doc.Close(0)
  if (-not $wordWasRunning) { $word.Quit() }
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
  "Wrote $pdf"
}
