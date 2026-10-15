param(
  [string]$Script = 'C:\StudyMaterialDoNotTouch\publications\URI-CADS dataset\presentation\Presentation_Script_Two_Presenters.docx',
  [string]$Pdf = 'C:\Users\anubh\AppData\Local\Temp\claude\C--StudyMaterialDoNotTouch-publications-URI-CADS-dataset\f3207856-96cc-4463-a75d-ed0cec1b22d9\scratchpad\qa\script.pdf',
  [int]$TimeoutSec = 150
)
# Word automation can block forever on an invisible dialog. Run it in a job with a hard timeout and
# clean up only Word processes started after this script began.
$start = Get-Date
$job = Start-Job -ScriptBlock {
  param($s, $p)
  $w = New-Object -ComObject Word.Application
  $w.Visible = $false
  $w.DisplayAlerts = 0
  $d = $w.Documents.Open($s, $false, $true, $false)
  $d.SaveAs2($p, 17)
  $pages = $d.ComputeStatistics(2)
  $d.Close(0)
  $w.Quit()
  "pages=$pages"
} -ArgumentList $Script, $Pdf

if (Wait-Job $job -Timeout $TimeoutSec) {
  Receive-Job $job
  "Wrote $Pdf"
} else {
  "TIMEOUT after $TimeoutSec s"
  Stop-Job $job
}
Remove-Job $job -Force
Get-Process WINWORD -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -ge $start -and -not $_.MainWindowTitle } | Stop-Process -Force
