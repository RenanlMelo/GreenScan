import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  FlatList,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import axios from "axios";
import { Ionicons } from "@expo/vector-icons";
import { styles } from "./Styles";
import { useReports } from "../../contexts/ReportContext";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/pt-br";
import { API_URL } from "../../constants/api";
import { useRoute } from "@react-navigation/native";

type RootStackParamList = {
  History: { openReportId?: number } | undefined;
  Report: { reportId: number };
};

type Props = NativeStackScreenProps<RootStackParamList, "History">;

export default function History({ navigation }: Props) {
  const { reports, setReports } = useReports();
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const route = useRoute();
  const params = route.params as { openReportId?: number } | undefined;

  useEffect(() => {
    fetchReports();
  }, []);

  // 🔁 Se vier da Home com um report específico, abre o relatório automaticamente
  useEffect(() => {
    if (params?.openReportId) {
      // pequeno delay para garantir que os dados já estejam carregados
      const timeout = setTimeout(() => {
        navigation.navigate("Report", { reportId: params.openReportId! });
      }, 300);
      return () => clearTimeout(timeout);
    }
  }, [params]);

  async function fetchReports() {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/reports/`);
      const data = response.data.data || response.data;
      // mostra os mais recentes primeiro
      setReports(data.slice().reverse());
    } catch (error: any) {
      console.error(
        "❌ Erro ao buscar reports:",
        error.response?.data || error.message
      );
    } finally {
      setLoading(false);
    }
  }

  function confirmDelete(id: number) {
    Alert.alert(
      "Deletar relatório",
      "Tem certeza que deseja deletar este relatório?",
      [
        { text: "Cancelar", style: "cancel" },
        {
          text: "Deletar",
          style: "destructive",
          onPress: () => handleDelete(id),
        },
      ]
    );
  }

  async function handleDelete(id: number) {
    try {
      setDeletingId(id);
      await axios.delete(`${API_URL}/reports/${id}`);
      setReports((prev) => prev.filter((r) => r.id !== id));
    } catch (error: any) {
      console.error(
        "❌ Erro ao deletar report:",
        error.response?.data || error.message
      );
      Alert.alert("Erro", "Não foi possível deletar o relatório.");
    } finally {
      setDeletingId(null);
    }
  }

  function openReport(id: number) {
    navigation.navigate("Report", { reportId: id });
  }

  // ✅ Configura o idioma do dayjs antes de calcular "há 2 dias", etc.
  dayjs.extend(relativeTime);
  dayjs.locale("pt-br");

  const renderItem = ({ item }: { item: any }) => {
    const timeAgo = dayjs(item.created_at).fromNow();
    const imageUri = item.image
      ? item.image.startsWith("http")
        ? item.image
        : `${API_URL}/${item.image.replace("\\", "/")}`
      : "https://via.placeholder.com/50";

    return (
      <TouchableOpacity onPress={() => openReport(item.id)} style={styles.item}>
        <Image style={styles.image} source={{ uri: imageUri }} />

        <View style={styles.info}>
          <Text style={styles.name}>{item.title}</Text>
          <Text numberOfLines={1} ellipsizeMode="tail" style={styles.situation}>
            {item.description}
          </Text>
        </View>

        <View style={styles.right}>
          <TouchableOpacity
            style={styles.deleteButton}
            onPress={() => confirmDelete(item.id)}
            disabled={deletingId === item.id}
          >
            {deletingId === item.id ? (
              <ActivityIndicator size="small" />
            ) : (
              <Ionicons name="trash-outline" size={22} color="#FF3B30" />
            )}
          </TouchableOpacity>
          <Text style={styles.time}>{timeAgo}</Text>
        </View>
      </TouchableOpacity>
    );
  };

  if (loading) {
    return (
      <View
        style={[
          styles.container,
          { justifyContent: "center", alignItems: "center" },
        ]}
      >
        <ActivityIndicator size="large" color="#00AA00" />
        <Text style={{ marginTop: 12 }}>Carregando histórico...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {reports.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyText}>Nenhum relatório encontrado.</Text>
        </View>
      ) : (
        <FlatList
          data={reports}
          keyExtractor={(item, index) => item.id.toString() + "_" + index}
          renderItem={renderItem}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={{ paddingBottom: 20 }}
        />
      )}
    </View>
  );
}
