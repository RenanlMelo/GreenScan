import React, { useEffect, useState } from "react";
import { View, Text, Image, ScrollView, ActivityIndicator } from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import axios from "axios";
import { styles } from "./Styles";
import { useReports } from "../../contexts/ReportContext";

type ScanStackParamList = {
  Report: { reportId: number };
};

type Props = NativeStackScreenProps<ScanStackParamList, "Report">;

type ReportData = {
  id: number;
  classe: string;
  confidence: number;
  title: string;
  description: string;
  treatment: string;
  prevention: string;
  image: string;
  created_at: string;
};

const API_URL = "https://greenscan-uak7.onrender.com";

export function Report({ route }: Props) {
  const { reportId } = route.params;
  const [report, setReport] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const { reports, addReport } = useReports(); // use para atualizar contexto após criação

  useEffect(() => {
    async function fetchReport() {
      try {
        const response = await axios.get(`${API_URL}/reports/${reportId}`);
        const data = response.data.data;
        setReport(data);

        // só adiciona se ainda não existir
        if (!reports.find((r) => r.id === data.id)) {
          addReport(data);
        }
      } catch (error) {
        console.error("Erro ao buscar relatório:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchReport();
  }, [reportId, reports]);

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
        <ActivityIndicator size="large" color="#00AA00" />
        <Text style={{ marginTop: 16 }}>Carregando relatório...</Text>
      </View>
    );
  }

  if (!report) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
        <Text>Não foi possível carregar o relatório.</Text>
      </View>
    );
  }

  const imageUri = report.image
    ? report.image.startsWith("https")
      ? report.image
      : `${API_URL}/${report.image.replace("\\", "/")}`
    : "https://via.placeholder.com/50";

  return (
    <ScrollView style={styles.containerReport}>
      <View style={styles.containerData}>
        <View style={styles.main}>
          <Image source={{ uri: imageUri }} style={styles.previewImage} />
          <View style={styles.responseTitle}>
            <Text style={styles.condition}>{report.title}</Text>
            <Text style={styles.title}>{report.classe}</Text>
          </View>
        </View>

        <View style={{ marginTop: 16 }}>
          <Text style={styles.sectionTitle}>Descrição:</Text>
          <Text style={styles.sectionText}>{report.description}</Text>

          <Text style={styles.sectionTitle}>Tratamento:</Text>
          <Text style={styles.sectionText}>{report.treatment}</Text>

          <Text style={styles.sectionTitle}>Prevenção:</Text>
          <Text style={styles.sectionText}>{report.prevention}</Text>
        </View>
      </View>
    </ScrollView>
  );
}
