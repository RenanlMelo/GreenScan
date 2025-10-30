import React, { useEffect, useState } from "react";
import { View, Text, Image, FlatList, ActivityIndicator } from "react-native";
import { styles } from "./Styles";
import { useReports, Report } from "../../contexts/ReportContext";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";

export function RecentHistory() {
  const { reports, setReports } = useReports();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchReports() {
      try {
        const response = await fetch(
          "https://greenscan-uak7.onrender.com/reports/"
        );
        const data = await response.json();

        // Ordena por created_at do mais recente para o mais antigo
        const sortedReports = (data.data || data)
          .sort(
            (a: Report, b: Report) =>
              new Date(b.created_at).getTime() -
              new Date(a.created_at).getTime()
          )
          .slice(0, 5); // <-- garante que realmente só vem 5

        setReports(sortedReports);
      } catch (error) {
        console.error("Erro ao buscar histórico:", error);
      } finally {
        setLoading(false);
      }
    }

    if (reports.length === 0) {
      fetchReports();
    } else {
      setLoading(false);
    }
  }, []);

  dayjs.extend(relativeTime);

  const renderItem = ({ item }: { item: Report }) => {
    const timeAgo = dayjs(item.created_at).fromNow();

    const imageUri = item.image
      ? item.image.startsWith("http")
        ? item.image
        : `https://greenscan-uak7.onrender.com/${item.image.replace("\\", "/")}`
      : "https://via.placeholder.com/50";

    return (
      <View style={styles.item}>
        <Image style={styles.image} source={{ uri: imageUri }} />
        <View style={styles.info}>
          <Text style={styles.name}>{item.title}</Text>
          <Text numberOfLines={1} ellipsizeMode="tail" style={styles.situation}>
            {item.description}
          </Text>
        </View>
        <Text style={styles.time}>{timeAgo}</Text>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
        <ActivityIndicator size="large" color="#00AA00" />
        <Text style={{ marginTop: 16 }}>Carregando histórico...</Text>
      </View>
    );
  }

  return (
    <FlatList
      data={reports.slice(0, 5)}
      keyExtractor={(item) => item.id.toString()}
      renderItem={renderItem}
      showsVerticalScrollIndicator={false}
      ListHeaderComponent={<Text style={styles.title}>Histórico Recente</Text>}
      contentContainerStyle={{ padding: 16, paddingBottom: 50 }}
    />
  );
}
