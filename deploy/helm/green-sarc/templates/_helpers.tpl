{{/* Expand the name of the chart. */}}
{{- define "green-sarc.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* A qualified app name. */}}
{{- define "green-sarc.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "green-sarc.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/* Common labels. */}}
{{- define "green-sarc.labels" -}}
app.kubernetes.io/name: {{ include "green-sarc.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{/* Selector labels. */}}
{{- define "green-sarc.selectorLabels" -}}
app.kubernetes.io/name: {{ include "green-sarc.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* Name of the secret holding auth token / API key. */}}
{{- define "green-sarc.secretName" -}}
{{- printf "%s-secrets" (include "green-sarc.fullname" .) -}}
{{- end -}}
