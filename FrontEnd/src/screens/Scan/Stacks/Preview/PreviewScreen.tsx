import React, { useEffect } from "react";
import { View, Text, ActivityIndicator } from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import axios from "axios";

type RootStackParamList = {
  Camera: undefined;
  Preview: { photoUri: string };
  Report: {
    photoUri: string;
    classe: string;
    confianca: number;
    tratamento: string;
  };
};

type Props = NativeStackScreenProps<RootStackParamList, "Preview">;

const API_URL = "http://192.168.15.182:8000";

export function PreviewScreen({ route, navigation }: Props) {
  const { photoUri } = route.params;

  useEffect(() => {
    sendPhoto(photoUri);
  }, []);

  async function sendPhoto(photoUri: string) {
    try {
      const formData = new FormData();
      formData.append("image", {
        uri: photoUri,
        name: "photo.jpg",
        type: "image/jpeg",
      } as any);

      const api = axios.create({ baseURL: API_URL });

      const classifyResponse = await api.postForm("/ai/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const aiData = classifyResponse.data;
      console.log("✅ IA classificou:", aiData);

      const reportPayload = {
        clss: aiData.classe,
        trust: aiData.confianca,
        treatment: aiData.tratamento,
      };

      await api.post("/reports/create", reportPayload);
      console.log("✅ Report criado no backend");

      navigation.replace("Report", {
        photoUri,
        classe: aiData.classe,
        confianca: aiData.confianca,
        tratamento: aiData.tratamento,
      });
    } catch (error: any) {
      console.error(
        "❌ Erro ao processar imagem/report:",
        error.response?.data || error.message
      );
    }
  }

  return (
    <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
      <ActivityIndicator size="large" color="#00AA00" />
      <Text style={{ marginTop: 16 }}>Processando imagem...</Text>
    </View>
  );
}
