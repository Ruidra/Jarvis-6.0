' ==========================================================================
'  Jarvis-Silent.vbs — start JARVIS with no console window at all.
'  Double-click this instead of Jarvis.bat if you dislike the black window.
' ==========================================================================
Option Explicit

Dim fso, sh, base, py, args
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")

base = fso.GetParentFolderName(WScript.ScriptFullName)

py = base & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(py) Then py = base & "\venv\Scripts\pythonw.exe"
If Not fso.FileExists(py) Then py = "pythonw.exe"

sh.CurrentDirectory = base
args = """" & py & """ """ & base & "\main.py"""
sh.Run args, 0, False
