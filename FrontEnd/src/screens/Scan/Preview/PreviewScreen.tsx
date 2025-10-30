import React, { useState } from "react";
import {
  View,
  Image,
  TouchableOpacity,
  Text,
  ActivityIndicator,
} from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import * as FileSystem from "expo-file-system/legacy";

type RootStackParamList = {
  Camera: undefined;
  Preview: { photoUri: string };
  Report: { reportId: number };
};

type Props = NativeStackScreenProps<RootStackParamList, "Preview">;

const API_URL = "https://greenscan-uak7.onrender.com";

export function PreviewScreen({ route, navigation }: Props) {
  const { photoUri } = route.params;
  const [loading, setLoading] = useState(false);

  async function fetchWithRetry(
    url: string,
    options: RequestInit,
    retries = 3,
    delay = 500
  ): Promise<Response> {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const response = await fetch(url, options);

        if (!response.ok) {
          const text = await response.text();
          throw new Error(`Erro ${response.status}: ${text}`);
        }

        return response; // sucesso ✅
      } catch (error) {
        console.warn(`Tentativa ${attempt} falhou:`, error);

        if (attempt === retries) {
          throw error; // todas falharam ❌
        }

        // espera antes da próxima tentativa
        await new Promise((res) => setTimeout(res, delay));
      }
    }

    throw new Error("Erro inesperado no fetchWithRetry");
  }

  async function sendPhoto() {
    try {
      setLoading(true);

      // Garante que o arquivo existe antes de enviar
      const fileInfo = await FileSystem.getInfoAsync(photoUri);
      if (!fileInfo.exists) {
        throw new Error("Arquivo da foto ainda não está disponível");
      }

      const formData = new FormData();
      formData.append("image", {
        uri: photoUri,
        name: "photo.jpg",
        type: "image/jpeg",
      } as any);

      // 🔁 Tenta até 3 vezes com 500ms de intervalo
      const response = await fetchWithRetry(`${API_URL}/ai/analyze`, {
        method: "POST",
        body: formData,
      });

      const reportData = await response.json();
      navigation.replace("Report", { reportId: reportData.data.id });
    } catch (error) {
      console.error("Erro ao enviar foto:", error);
    } finally {
      setLoading(false);
    }
  }

  return (
    <View style={{ flex: 1, backgroundColor: "black" }}>
      {/* Imagem ocupando toda a tela */}
      <Image
        source={{ uri: photoUri }}
        style={{ flex: 1, resizeMode: "contain" }}
      />

      {/* Botões */}
      <View
        style={{
          position: "absolute",
          bottom: 32,
          width: "100%",
          flexDirection: "row",
          justifyContent: "space-around",
          alignItems: "center",
        }}
      >
        {/* Tirar outra */}
        <TouchableOpacity
          style={{
            width: 75,
            height: 75,
            borderRadius: 50,
            backgroundColor: "#303030aa",
            justifyContent: "center",
            alignItems: "center",
          }}
          onPress={() => navigation.goBack()}
        >
          <MaterialCommunityIcons
            name="camera-retake"
            size={32}
            color="white"
          />
        </TouchableOpacity>

        {/* Confirmar */}
        <TouchableOpacity
          style={{
            width: 70,
            height: 70,
            borderRadius: 35,
            backgroundColor: "#00AA00",
            justifyContent: "center",
            alignItems: "center",
          }}
          onPress={sendPhoto}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator size="small" color="white" />
          ) : (
            <MaterialCommunityIcons name="check" size={36} color="white" />
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}
